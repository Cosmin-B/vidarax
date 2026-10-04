import importlib.util
import socket
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


spec = importlib.util.spec_from_file_location(
    "embedding_resources_server", Path(__file__).with_name("embedding_server.py")
)
server = importlib.util.module_from_spec(spec)
# Exercise ownership without loading a model or numerical libraries.
with mock.patch.dict(sys.modules, {
    "embedding_resources_server": server,
    "numpy": mock.MagicMock(),
    "torch": SimpleNamespace(inference_mode=lambda: lambda function: function),
    "PIL": SimpleNamespace(Image=mock.Mock()),
}):
    spec.loader.exec_module(server)


class EmbeddingPayloadLifetimeTests(unittest.TestCase):
    def test_full_queue_releases_the_untransferred_payload(self):
        batcher = server.MicroBatcher(1, 8, 1, 0)
        queued = server.WorkItem(bytearray(b"jpeg"))
        rejected = server.WorkItem(bytearray(b"jpeg"))
        self.assertTrue(batcher.reserve(4))
        self.assertTrue(batcher.submit_reserved(queued))
        self.assertTrue(batcher.reserve(4))
        self.assertFalse(batcher.submit_reserved(rejected))
        self.assertEqual(rejected.jpeg, bytearray())
        self.assertEqual(queued.jpeg, bytearray(b"jpeg"))
        self.assertEqual(batcher.queued_bytes, 4)

    def test_completed_batch_releases_bytes_retained_by_a_persistent_handler(self):
        batcher = server.MicroBatcher(1, 4, 1, 0)
        item = server.WorkItem(bytearray(b"jpeg"))
        self.assertTrue(batcher.reserve(4))
        batcher.work = mock.Mock()
        batcher.work.get.side_effect = [item, StopIteration]

        def infer(batch):
            self.assertEqual(batch[0].jpeg, b"jpeg")
            self.assertEqual(batcher.queued_bytes, 4)
            batch[0].embedding = b"embedding"
            batch[0].done.set()

        batcher._infer = infer
        with self.assertRaises(StopIteration):
            batcher._run()
        self.assertTrue(item.done.is_set())
        self.assertEqual(item.embedding, b"embedding")
        self.assertEqual(item.jpeg, bytearray())
        self.assertEqual(batcher.queued_bytes, 0)
        self.assertTrue(batcher.reserve(4))

    def test_failed_batch_releases_transferred_payload_and_budget(self):
        batcher = server.MicroBatcher(2, 8, 1, 0)
        item = server.WorkItem(bytearray(b"jpeg"))
        self.assertTrue(batcher.reserve(4))
        next_item = server.WorkItem(bytearray(b"next"))
        self.assertTrue(batcher.reserve(4))
        batcher.work = mock.Mock()
        batcher.work.get.side_effect = [item, next_item, StopIteration]
        batcher._infer = mock.Mock(side_effect=RuntimeError("failed"))
        with self.assertLogs(server.logger, level="ERROR"):
            with self.assertRaises(StopIteration):
                batcher._run()
        self.assertEqual(item.jpeg, bytearray())
        self.assertTrue(item.done.is_set())
        self.assertEqual(item.error, "embedding batch failed")
        self.assertTrue(next_item.done.is_set())
        self.assertEqual(batcher._infer.call_count, 2)
        self.assertEqual(batcher.queued_bytes, 0)

    def test_partial_progress_cannot_restart_payload_deadline(self):
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

    def test_response_body_uses_time_left_after_header(self):
        handler = server.EmbeddingRequestHandler.__new__(server.EmbeddingRequestHandler)
        handler.request = mock.Mock()
        handler.deadline = 1.0
        with mock.patch.object(server.time, "perf_counter", side_effect=[0.4, 1.1]):
            with self.assertRaises(socket.timeout):
                handler._send_response(b"header", b"embedding")
        handler.request.sendall.assert_called_once_with(b"header")


if __name__ == "__main__":
    unittest.main()
