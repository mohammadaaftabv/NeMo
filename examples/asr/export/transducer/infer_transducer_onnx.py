# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import json
import os
import tempfile
from argparse import ArgumentParser

import torch
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.submodules.rnnt_greedy_decoding import ONNXGreedyBatchedRNNTInfer
from nemo.utils import logging
import onnx
import onnxruntime

"""
Script to compare the outputs of a NeMo Pytorch based RNNT Model and its ONNX exported representation.

# Compare a NeMo and ONNX model
python infer_transducer_onnx.py \
    --nemo_model="<path to a .nemo file>" \
    OR
    --pretrained_model="<name of a pretrained model>" \
    --onnx_encoder="<path to onnx encoder file>" \
    --onnx_decoder="<path to onnx decoder-joint file>" \
    --dataset_manifest="<Either pass a manifest file path here>" \
    --audio_dir="<Or pass a directory containing preprocessed monochannel audio files>" \
    --max_symbold_per_step=5 \
    --batch_size=32 \
    --log
    
# Export and compare a NeMo and ONNX model
python infer_transducer_onnx.py \
    --nemo_model="<path to a .nemo file>" \
    OR
    --pretrained_model="<name of a pretrained model>" \
    --export \
    --dataset_manifest="<Either pass a manifest file path here>" \
    --audio_dir="<Or pass a directory containing preprocessed monochannel audio files>" \
    --max_symbold_per_step=5 \
    --batch_size=32 \
    --log
"""

@dataclass
class Hypothesis:
    """Hypothesis class for beam search algorithms.

    score: A float score obtained from an AbstractRNNTDecoder module's score_hypothesis method.

    y_sequence: Either a sequence of integer ids pointing to some vocabulary, or a packed torch.Tensor
        behaving in the same manner. dtype must be torch.Long in the latter case.

    dec_state: A list (or list of list) of LSTM-RNN decoder states. Can be None.

    text: (Optional) A decoded string after processing via CTC / RNN-T decoding (removing the CTC/RNNT
        `blank` tokens, and optionally merging word-pieces). Should be used as decoded string for
        Word Error Rate calculation.

    timestep: (Optional) A list of integer indices representing at which index in the decoding
        process did the token appear. Should be of same length as the number of non-blank tokens.

    alignments: (Optional) Represents the CTC / RNNT token alignments as integer tokens along an axis of
        time T (for CTC) or Time x Target (TxU).
        For CTC, represented as a single list of integer indices.
        For RNNT, represented as a dangling list of list of integer indices.
        Outer list represents Time dimension (T), inner list represents Target dimension (U).
        The set of valid indices **includes** the CTC / RNNT blank token in order to represent alignments.

    frame_confidence: (Optional) Represents the CTC / RNNT per-frame confidence scores as token probabilities
        along an axis of time T (for CTC) or Time x Target (TxU).
        For CTC, represented as a single list of float indices.
        For RNNT, represented as a dangling list of list of float indices.
        Outer list represents Time dimension (T), inner list represents Target dimension (U).

    token_confidence: (Optional) Represents the CTC / RNNT per-token confidence scores as token probabilities
        along an axis of Target U.
        Represented as a single list of float indices.

    word_confidence: (Optional) Represents the CTC / RNNT per-word confidence scores as token probabilities
        along an axis of Target U.
        Represented as a single list of float indices.

    length: Represents the length of the sequence (the original length without padding), otherwise
        defaults to 0.

    y: (Unused) A list of torch.Tensors representing the list of hypotheses.

    lm_state: (Unused) A dictionary state cache used by an external Language Model.

    lm_scores: (Unused) Score of the external Language Model.

    tokens: (Optional) A list of decoded tokens (can be characters or word-pieces.

    last_token (Optional): A token or batch of tokens which was predicted in the last step.
    """

    score: float
    y_sequence: Union[List[int], torch.Tensor]
    text: Optional[str] = None
    dec_out: Optional[List[torch.Tensor]] = None
    dec_state: Optional[Union[List[List[torch.Tensor]], List[torch.Tensor]]] = None
    timestep: Union[List[int], torch.Tensor] = field(default_factory=list)
    alignments: Optional[Union[List[int], List[List[int]]]] = None
    frame_confidence: Optional[Union[List[float], List[List[float]]]] = None
    token_confidence: Optional[List[float]] = None
    word_confidence: Optional[List[float]] = None
    length: Union[int, torch.Tensor] = 0
    y: List[torch.tensor] = None
    lm_state: Optional[Union[Dict[str, Any], List[Any]]] = None
    lm_scores: Optional[torch.Tensor] = None
    tokens: Optional[Union[List[int], torch.Tensor]] = None
    last_token: Optional[torch.Tensor] = None

    @property
    def non_blank_frame_confidence(self) -> List[float]:
        """Get per-frame confidence for non-blank tokens according to self.timestep

        Returns:
            List with confidence scores. The length of the list is the same as `timestep`.
        """
        non_blank_frame_confidence = []
        # self.timestep can be a dict for RNNT
        timestep = self.timestep['timestep'] if isinstance(self.timestep, dict) else self.timestep
        if len(self.timestep) != 0 and self.frame_confidence is not None:
            if any(isinstance(i, list) for i in self.frame_confidence):  # rnnt
                t_prev = -1
                offset = 0
                for t in timestep:
                    if t != t_prev:
                        t_prev = t
                        offset = 0
                    else:
                        offset += 1
                    non_blank_frame_confidence.append(self.frame_confidence[t][offset])
            else:  # ctc
                non_blank_frame_confidence = [self.frame_confidence[t] for t in timestep]
        return non_blank_frame_confidence

    @property
    def words(self) -> List[str]:
        """Get words from self.text

        Returns:
            List with words (str).
        """
        return [] if self.text is None else self.text.split()



def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        "--nemo_model", type=str, default=None, required=False, help="Path to .nemo file",
    )
    parser.add_argument(
        '--pretrained_model', type=str, default=None, required=False, help='Name of a pretrained NeMo file'
    )
    parser.add_argument('--onnx_encoder', type=str, default=None, required=False, help="Path to onnx encoder model")
    parser.add_argument(
        '--onnx_decoder', type=str, default=None, required=False, help="Path to onnx decoder + joint model"
    )
    parser.add_argument('--threshold', type=float, default=0.01, required=False)

    parser.add_argument('--dataset_manifest', type=str, default=None, required=False, help='Path to dataset manifest')
    parser.add_argument('--audio_dir', type=str, default=None, required=False, help='Path to directory of audio files')
    parser.add_argument('--audio_type', type=str, default='wav', help='File format of audio')

    parser.add_argument('--export', action='store_true', help="Whether to export the model into onnx prior to eval")
    parser.add_argument('--max_symbold_per_step', type=int, default=5, required=False, help='Number of decoding steps')
    parser.add_argument('--batch_size', type=int, default=32, help='Batchsize')
    parser.add_argument('--log', action='store_true', help='Log the predictions between pytorch and onnx')

    args = parser.parse_args()
    return args


def assert_args(args):
    if args.nemo_model is None and args.pretrained_model is None:
        raise ValueError(
            "`nemo_model` or `pretrained_model` must be passed ! It is required for decoding the RNNT tokens "
            "and ensuring predictions match between Torch and ONNX."
        )

    if args.nemo_model is not None and args.pretrained_model is not None:
        raise ValueError(
            "`nemo_model` and `pretrained_model` cannot both be passed ! Only one can be passed to this script."
        )

    if args.export and (args.onnx_encoder is not None or args.onnx_decoder is not None):
        raise ValueError("If `export` is set, then `onnx_encoder` and `onnx_decoder` arguments must be None")

    if args.audio_dir is None and args.dataset_manifest is None:
        raise ValueError("Both `dataset_manifest` and `audio_dir` cannot be None!")

    if args.audio_dir is not None and args.dataset_manifest is not None:
        raise ValueError("Submit either `dataset_manifest` or `audio_dir`.")

    if int(args.max_symbold_per_step) < 1:
        raise ValueError("`max_symbold_per_step` must be an integer > 0")


def export_model_if_required(args, nemo_model):
    if args.export:
        nemo_model.export("temp_rnnt.onnx")
        args.onnx_encoder = "encoder-temp_rnnt.onnx"
        args.onnx_decoder = "decoder_joint-temp_rnnt.onnx"


def resolve_audio_filepaths(args):
    # get audio filenames
    if args.audio_dir is not None:
        filepaths = list(glob.glob(os.path.join(args.audio_dir, f"*.{args.audio_type}")))
    else:
        # get filenames from manifest
        filepaths = []
        with open(args.dataset_manifest, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                filepaths.append(item['audio_filepath'])

    logging.info(f"\nTranscribing {len(filepaths)} files...\n")

    return filepaths


def main():

    def setup_blank_index():
        # ASSUME: Single input with no time length information
        global blank_index
        dynamic_dim = 257
        shapes = encoder_inputs[0].type.tensor_type.shape.dim
        ip_shape = []
        for shape in shapes:
            if hasattr(shape, 'dim_param') and 'dynamic' in shape.dim_param:
                ip_shape.append(dynamic_dim)  # replace dynamic axes with constant
            else:
                ip_shape.append(int(shape.dim_value))

        enc_logits, encoded_length = run_encoder(
            audio_signal=torch.randn(*ip_shape), length=torch.randint(0, 1, size=(dynamic_dim,))
        )

        # prepare states
        states = get_initial_states(batchsize=dynamic_dim)

        # run decoder 1 step
        joint_out, states = run_decoder(enc_logits, None, None, *states)
        log_probs, lengths = joint_out

        blank_index = log_probs.shape[-1] - 1  # last token of vocab size is blank token
        return blank_index
        logging.info(
            f"Enc-Dec-Joint step was evaluated, blank token id = {blank_index}; vocab size = {log_probs.shape[-1]}"
        )

    def run_encoder(audio_signal, length):
        if hasattr(audio_signal, 'cpu'):
            audio_signal = audio_signal.cpu().numpy()

        if hasattr(length, 'cpu'):
            length = length.cpu().numpy()

        ip = {
            encoder_inputs[0].name: audio_signal,
            encoder_inputs[1].name: length,
        }
        enc_out = encoder.run(None, ip)
        enc_out, encoded_length = enc_out  # ASSUME: single output
        return enc_out, encoded_length

    def run_decoder(enc_logits, targets, target_length, *states):
        # ASSUME: Decoder is RNN Transducer
        if targets is None:
            targets = torch.zeros(enc_logits.shape[0], 1, dtype=torch.int32)
            target_length = torch.ones(enc_logits.shape[0], dtype=torch.int32)

        if hasattr(targets, 'cpu'):
            targets = targets.cpu().numpy()

        if hasattr(target_length, 'cpu'):
            target_length = target_length.cpu().numpy()

        ip = {
            decoder_inputs[0].name: enc_logits,
            decoder_inputs[1].name: targets,
            decoder_inputs[2].name: target_length,
        }

        num_states = 0
        if states is not None and len(states) > 0:
            num_states = len(states)
            for idx, state in enumerate(states):
                if hasattr(state, 'cpu'):
                    state = state.cpu().numpy()

                ip[decoder_inputs[len(ip)].name] = state

        dec_out = decoder.run(None, ip)

        # unpack dec output
        if num_states > 0:
            new_states = dec_out[-num_states:]
            dec_out = dec_out[:-num_states]
        else:
            new_states = None

        return dec_out, new_states

    def get_initial_states(batchsize):
        # ASSUME: LSTM STATES of shape (layers, batchsize, dim)
        input_state_nodes = [ip for ip in decoder_inputs if 'state' in ip.name]
        num_states = len(input_state_nodes)
        if num_states == 0:
            return

        input_states = []
        for state_id in range(num_states):
            node = input_state_nodes[state_id]
            ip_shape = []
            for shape_idx, shape in enumerate(node.type.tensor_type.shape.dim):
                if hasattr(shape, 'dim_param') and 'dynamic' in shape.dim_param:
                    ip_shape.append(batchsize)  # replace dynamic axes with constant
                else:
                    ip_shape.append(int(shape.dim_value))

            input_states.append(torch.zeros(*ip_shape))

        return input_states

    def greedy_decode(x, out_len):
        # x: [B, T, D]
        # out_len: [B]

        # Initialize state
        batchsize = x.shape[0]
        hidden = get_initial_states(batchsize)
        target_lengths = torch.ones(batchsize, dtype=torch.int32)

        # Output string buffer
        label = [[] for _ in range(batchsize)]
        timesteps = [[] for _ in range(batchsize)]

        # Last Label buffer + Last Label without blank buffer
        # batch level equivalent of the last_label
        print([batchsize, 1], blank_index, 'duplic')
        last_label = torch.full([batchsize, 1], fill_value=blank_index, dtype=torch.long).numpy()
        if torch.is_tensor(x):
            last_label = torch.from_numpy(last_label).to(device)

        # Mask buffers
        blank_mask = torch.full([batchsize], fill_value=0, dtype=torch.bool).numpy()

        # Get max sequence length
        max_out_len = out_len.max()
        for time_idx in range(max_out_len):
            f = x[:, time_idx : time_idx + 1, :]  # [B, 1, D]

            if torch.is_tensor(f):
                f = f.transpose(1, 2)
            else:
                f = f.transpose([0, 2, 1])

            # Prepare t timestamp batch variables
            not_blank = True
            symbols_added = 0

            # Reset blank mask
            blank_mask *= False

            # Update blank mask with time mask
            # Batch: [B, T, D], but Bi may have seq len < max(seq_lens_in_batch)
            # Forcibly mask with "blank" tokens, for all sample where current time step T > seq_len
            blank_mask = time_idx >= out_len
            # Start inner loop
            while not_blank and (max_symbols_per_step is None or symbols_added < max_symbols_per_step):

                # Batch prediction and joint network steps
                # If very first prediction step, submit SOS tag (blank) to pred_step.
                # This feeds a zero tensor as input to AbstractRNNTDecoder to prime the state
                if time_idx == 0 and symbols_added == 0:
                    g = torch.tensor([blank_index] * batchsize, dtype=torch.int32).view(-1, 1)
                else:
                    if torch.is_tensor(last_label):
                        g = last_label.type(torch.int32)
                    else:
                        g = last_label.astype(np.int32)

                # Batched joint step - Output = [B, V + 1]
                joint_out, hidden_prime = run_decoder(f, g, target_lengths, *hidden)
                logp, pred_lengths = joint_out
                logp = logp[:, 0, 0, :]

                # Get index k, of max prob for batch
                if torch.is_tensor(logp):
                    v, k = logp.max(1)
                else:
                    k = np.argmax(logp, axis=1).astype(np.int32)

                # Update blank mask with current predicted blanks
                # This is accumulating blanks over all time steps T and all target steps min(max_symbols, U)
                k_is_blank = k == blank_index
                blank_mask |= k_is_blank

                del k_is_blank
                del logp

                # If all samples predict / have predicted prior blanks, exit loop early
                # This is equivalent to if single sample predicted k
                if blank_mask.all():
                    not_blank = False

                else:
                    # Collect batch indices where blanks occurred now/past
                    if torch.is_tensor(blank_mask):
                        blank_indices = (blank_mask == 1).nonzero(as_tuple=False)
                    else:
                        blank_indices = blank_mask.astype(np.int32).nonzero()

                    if type(blank_indices) in (list, tuple):
                        blank_indices = blank_indices[0]

                    # Recover prior state for all samples which predicted blank now/past
                    if hidden is not None:
                        # LSTM has 2 states
                        for state_id in range(len(hidden)):
                            hidden_prime[state_id][:, blank_indices, :] = hidden[state_id][:, blank_indices, :]

                    elif len(blank_indices) > 0 and hidden is None:
                        # Reset state if there were some blank and other non-blank predictions in batch
                        # Original state is filled with zeros so we just multiply
                        # LSTM has 2 states
                        for state_id in range(len(hidden_prime)):
                            hidden_prime[state_id][:, blank_indices, :] *= 0.0

                    # Recover prior predicted label for all samples which predicted blank now/past
                    k[blank_indices] = last_label[blank_indices, 0]

                    # Update new label and hidden state for next iteration
                    if torch.is_tensor(k):
                        last_label = k.clone().reshape(-1, 1)
                    else:
                        last_label = k.copy().reshape(-1, 1)
                    hidden = hidden_prime

                    # Update predicted labels, accounting for time mask
                    # If blank was predicted even once, now or in the past,
                    # Force the current predicted label to also be blank
                    # This ensures that blanks propogate across all timesteps
                    # once they have occured (normally stopping condition of sample level loop).
                    for kidx, ki in enumerate(k):
                        if blank_mask[kidx] == 0:
                            label[kidx].append(ki)
                            timesteps[kidx].append(time_idx)

                    symbols_added += 1

        return label, timesteps

    
    args = parse_arguments()

    providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    onnx_session_opt = onnxruntime.SessionOptions()
    onnx_session_opt.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Instantiate pytorch model
    if args.nemo_model is not None:
        nemo_model = args.nemo_model
        nemo_model = ASRModel.restore_from(nemo_model, map_location=device)  # type: ASRModel
        nemo_model.freeze()
    elif args.pretrained_model is not None:
        nemo_model = args.pretrained_model
        nemo_model = ASRModel.from_pretrained(nemo_model, map_location=device)  # type: ASRModel
        nemo_model.freeze()
    else:
        raise ValueError("Please pass either `nemo_model` or `pretrained_model` !")

    if torch.cuda.is_available():
        nemo_model = nemo_model.to('cuda')

    export_model_if_required(args, nemo_model)

    # Instantiate RNNT Decoding loop
    encoder_model_path='/home/aaftabv/riva_rnnt/encoder-temp_rnnt.onnx'
    decoder_model_path='/home/aaftabv/riva_rnnt/decoder_joint-temp_rnnt.onnx'

    onnx_model = onnx.load(encoder_model_path)
    onnx.checker.check_model(onnx_model, full_check=True)

    encoder_model = onnx_model
    encoder = onnxruntime.InferenceSession(
            onnx_model.SerializeToString(), providers=providers, provider_options=onnx_session_opt
        )

    onnx_model = onnx.load(decoder_model_path)
    onnx.checker.check_model(onnx_model, full_check=True)

    decoder_model = onnx_model
    decoder = onnxruntime.InferenceSession(
            onnx_model.SerializeToString(), providers=providers, provider_options=onnx_session_opt
        )

    encoder_inputs = list(encoder_model.graph.input)
    encoder_outputs = list(encoder_model.graph.output)
    decoder_inputs = list(decoder_model.graph.input)
    decoder_outputs = list(decoder_model.graph.output)

    blank_index=setup_blank_index()
    max_symbols_per_step = args.max_symbold_per_step
    decoding = ONNXGreedyBatchedRNNTInfer(encoder_model_path, decoder_model_path, max_symbols_per_step)

    audio_filepath = resolve_audio_filepaths(args)

    # Evaluate Pytorch Model (CPU/GPU)
    actual_transcripts = nemo_model.transcribe(audio_filepath, batch_size=args.batch_size)[0]

    # Evaluate ONNX model
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, 'manifest.json'), 'w', encoding='utf-8') as fp:
            for audio_file in audio_filepath:
                entry = {'audio_filepath': audio_file, 'duration': 100000, 'text': 'nothing'}
                fp.write(json.dumps(entry) + '\n')

        config = {'paths2audio_files': audio_filepath, 'batch_size': args.batch_size, 'temp_dir': tmpdir}

        nemo_model.preprocessor.featurizer.dither = 0.0
        nemo_model.preprocessor.featurizer.pad_to = 0

        temporary_datalayer = nemo_model._setup_transcribe_dataloader(config)

        all_hypothesis = []
        all_hypothesis_unpacked = []
        
        for test_batch in tqdm(temporary_datalayer, desc="ONNX Transcribing"):
            input_signal, input_signal_length = test_batch[0], test_batch[1]
            input_signal = input_signal.to(device)
            input_signal_length = input_signal_length.to(device)

            # Acoustic features
            processed_audio, processed_audio_len = nemo_model.preprocessor(
                input_signal=input_signal, length=input_signal_length
            )
            print(processed_audio,processed_audio.shape,processed_audio_len)
            # RNNT Decoding loop
            hypotheses_func = decoding(audio_signal=processed_audio, length=processed_audio_len)
            #RNNT Decoding without NeMo package dependency
            with torch.no_grad():
                # Apply optional preprocessing

                encoder_output, encoded_lengths = run_encoder(audio_signal=processed_audio, length=processed_audio_len)
                if torch.is_tensor(encoder_output):
                    encoder_output = encoder_output.transpose(1, 2)
                else:
                    encoder_output = encoder_output.transpose([0, 2, 1])  # (B, T, D)
                logitlen = encoded_lengths

                inseq = encoder_output  # [B, T, D]
                hypotheses, timestamps = greedy_decode(inseq, logitlen)

                # Pack the hypotheses results
                packed_result = [Hypothesis(score=-1.0, y_sequence=[]) for _ in range(len(hypotheses))]
                for i in range(len(packed_result)):
                    packed_result[i].y_sequence = torch.tensor(hypotheses[i], dtype=torch.long)
                    packed_result[i].length = timestamps[i]
            del hypotheses
            unpacked_hypotheses=packed_result
            
            # Process hypothesis (map char/subword token ids to text)
            hypotheses_func = nemo_model.decoding.decode_hypothesis(hypotheses_func)  # type: List[str]

            # Extract text from the hypothesis
            texts = [h.text for h in hypotheses_func]
            
            all_hypothesis += texts
            unpacked_hypotheses = nemo_model.decoding.decode_hypothesis(unpacked_hypotheses)  # type: List[str]

            # Extract text from the hypothesis
            texts = [h.text for h in unpacked_hypotheses]
            
            all_hypothesis_unpacked += texts
            del processed_audio, processed_audio_len
            del test_batch
    
    if args.log:
        for pt_transcript, onnx_transcript, defunc_transcript in zip(actual_transcripts, all_hypothesis,all_hypothesis_unpacked):
            print(f"Pytorch Transcripts : {pt_transcript}")
            print(f"ONNX Transcripts    : {onnx_transcript}")
            print(f"ONNX non-nemo dependent Transcripts    : {defunc_transcript}")
        print()

    # Measure error rate between onnx and pytorch transcipts
    pt_onnx_cer = word_error_rate(all_hypothesis, actual_transcripts, use_cer=True)
    assert pt_onnx_cer < args.threshold, "Threshold violation !"

    print("Character error rate between Pytorch and ONNX :", pt_onnx_cer)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
