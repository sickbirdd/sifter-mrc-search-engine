import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from unittest import TestCase, main
from modules.utils.logger import Test, logging
from starlette.testclient import TestClient
from modules.mrc_service.server import app
import asyncio

LOGGER = logging.getLogger('test')
client = TestClient(app)

class ServerTest(TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @Test("테스트지롱")
    def test_read(self):
        response = client.get("/inference", params={"question" : "손흥민 이번 시즌 성적 알려주세요."})
        print(response)
        assert response.status_code == 200

if __name__ == '__main__':
    main()