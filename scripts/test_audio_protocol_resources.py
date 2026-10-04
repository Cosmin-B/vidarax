import importlib.util
import io
import socket
import sys
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


# Framing and admission do not use numerical operations or MessagePack here.
# Keep these checks independent of model and numerical dependency installation.
spec = importlib.util.spec_from_file_location(
    "audio_resources_server", Path(__file__).with_name("audio_perception_server.py")
)
server = importlib.util.module_from_spec(spec)
with mock.patch.dict(
    sys.modules,
    {"audio_resources_server": server, "numpy": mock.MagicMock(), "msgpack": mock.Mock()},
):
    spec.loader.exec_module(server)


class AudioPayloadAdmissionTests(unittest.TestCase):
    def handler(self, admission):
        handler = server.AudioRequestHandler.__new__(server.AudioRequestHandler)
        handler.request = mock.Mock()
        handler.server = SimpleNamespace(
            admission=admission,
            request_timeout_s=0,
            engine=mock.Mock(),
        )
        handler.server.engine.analyze.return_value = {}
        handler._respond = mock.Mock()
        handler._respond_error = mock.Mock()
        return handler

    def header(self):
        return server.REQUEST_HEADER.pack(
            server.REQUEST_MAGIC, server.PROTOCOL_VERSION, server.OP_ANALYZE,
            0, 0, 0, 0, 0, server.MAX_AUDIO_BYTES, 0, 32, 3500,
        )

    def test_overload_rejects_before_payload_allocation(self):
        admission = server.Admission(1, 0)
        admission.acquire(0)
        handler = self.handler(admission)
        try:
            with mock.patch.object(server, "_read_exact", return_value=self.header()) as read:
                handler.handle()
            read.assert_called_once_with(handler.request, server.REQUEST_HEADER.size, mock.ANY)
            self.assertEqual(handler._respond_error.call_args.args[0], server.STATUS_OVERLOADED)
            handler.server.engine.analyze.assert_not_called()
            self.assertEqual(admission.snapshot(), {"active": 1, "queued": 0, "limit": 1})
        finally:
            admission.release()

    def test_queue_timeout_leaves_no_payload_or_queue_reservation(self):
        admission = server.Admission(1, 1)
        admission.acquire(0)
        handler = self.handler(admission)
        try:
            with mock.patch.object(server, "_read_exact", return_value=self.header()) as read:
                handler.handle()
            self.assertEqual(read.call_count, 1)
            self.assertEqual(handler._respond_error.call_args.args[1], "timeout")
            self.assertEqual(admission.snapshot(), {"active": 1, "queued": 0, "limit": 1})
        finally:
            admission.release()

    def test_truncated_payload_and_read_timeout_release_admission(self):
        for body in [None, socket.timeout("partial body")]:
            with self.subTest(body=body):
                admission = server.Admission(1, 0)
                handler = self.handler(admission)
                with mock.patch.object(server, "_read_exact", side_effect=[self.header(), body, b""]):
                    handler.handle()
                handler.server.engine.analyze.assert_not_called()
                self.assertEqual(admission.snapshot(), {"active": 0, "queued": 0, "limit": 1})
                self.assertEqual(admission.acquire(0), 0)
                admission.release()

    def test_payload_and_response_share_one_admission_owner(self):
        admission = server.Admission(1, 0)
        handler = self.handler(admission)
        reads = iter([self.header(), b"pcm", b"", None])

        def read(_socket, length, deadline):
            if length != server.REQUEST_HEADER.size:
                self.assertEqual(admission.snapshot()["active"], 1)
            return next(reads)

        def respond(metadata, audio):
            self.assertEqual(admission.snapshot()["active"], 1)

        handler._respond.side_effect = respond
        with mock.patch.object(server, "_read_exact", side_effect=read):
            handler.handle()
        handler.server.engine.analyze.assert_called_once()
        self.assertEqual(admission.snapshot(), {"active": 0, "queued": 0, "limit": 1})

    def test_partial_progress_does_not_extend_payload_deadline(self):
        sock = mock.Mock()

        def receive(view):
            view[0] = 0
            return 1

        sock.recv_into.side_effect = receive
        with mock.patch.object(server.time, "perf_counter", side_effect=[0.0, 0.4, 1.1]):
            with self.assertRaises(socket.timeout):
                server._read_exact(sock, 3, 1.0)
        self.assertEqual(sock.recv_into.call_count, 2)
        self.assertEqual(sock.settimeout.call_args_list, [mock.call(1.0), mock.call(0.6)])

    def test_queue_wait_uses_remaining_request_budget(self):
        admission = mock.Mock(spec=server.Admission)
        admission.acquire.return_value = 0
        admission.snapshot.return_value = {"active": 1, "queued": 0, "limit": 1}
        handler = self.handler(admission)
        handler.server.request_timeout_s = 1.0
        with mock.patch.object(server.time, "perf_counter", side_effect=[0.0, 0.75, 2.0]):
            with mock.patch.object(server, "_read_exact", side_effect=[self.header(), b"pcm", b"", None]):
                handler.handle()
        admission.acquire.assert_called_once_with(0.25)
        admission.release.assert_called_once()


class AudioDecodedCapacityTests(unittest.TestCase):
    def setUp(self):
        server.np.reset_mock()

    def wav(self, frames, sample_rate=1):
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(b"\0\0" * frames)
        return output.getvalue()

    def test_low_sample_rate_cannot_expand_past_the_audio_window(self):
        with self.assertRaises(server.DecodeError):
            server._decode_pcm_wav(self.wav(61))
        server.np.frombuffer.assert_not_called()

    def test_sixty_second_boundary_retains_arbitrary_sample_rates(self):
        for sample_rate in [1, 16000, 32000]:
            with self.subTest(sample_rate=sample_rate):
                _, rate = server._decode_pcm_wav(self.wav(60 * sample_rate, sample_rate))
                self.assertEqual(rate, sample_rate)

    def test_truncated_finite_frame_count_is_rejected_before_conversion(self):
        data = bytearray(self.wav(10))
        data[40:44] = (22).to_bytes(4, "little")
        with self.assertRaises(server.DecodeError):
            server._decode_pcm_wav(bytes(data))
        server.np.frombuffer.assert_not_called()

    def test_streaming_wav_unknown_length_is_accepted_with_the_same_bound(self):
        data = bytearray(self.wav(60))
        data[4:8] = (0xffffffff).to_bytes(4, "little")
        data[40:44] = (0xffffffff).to_bytes(4, "little")
        _, rate = server._decode_pcm_wav(bytes(data))
        self.assertEqual(rate, 1)

        server.np.reset_mock()
        data.extend(b"\0\0")
        with self.assertRaises(server.DecodeError):
            server._decode_pcm_wav(bytes(data))
        server.np.frombuffer.assert_not_called()

    def test_incomplete_pcm_frame_is_rejected_before_conversion(self):
        data = bytearray(self.wav(10))
        data[40:44] = (21).to_bytes(4, "little")
        data.append(0)
        data[4:8] = (len(data) - 8).to_bytes(4, "little")
        with self.assertRaises(server.DecodeError):
            server._decode_pcm_wav(bytes(data))
        server.np.frombuffer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
