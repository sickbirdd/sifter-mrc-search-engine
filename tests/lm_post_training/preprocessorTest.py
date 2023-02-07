import os
import sys
import yaml

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from modules.lm_post_training.preprocessor import Preprocessor as pp
from unittest import TestCase, main
from modules.config.logging import Test, logging

class preProcessorTest(TestCase):
    
    # 클래스 생성시 한번만 실행
    @classmethod
    def setUpClass(self):
        # 설정 파일 만들어지면 관련 변수로 대체할 것
        with open('modules/config.yaml') as f:
            conf = yaml.safe_load(f)
        self.modelName = conf["model"]["name"]
        self.implPreProcessor = pp(self.modelName)
        self.dataPath = conf["dataset"]["post_training"]["test"]["path"]
        self.dataDom = conf["dataset"]["post_training"]["test"]["struct"].split('/')
        
        logging.info("1:---최초 생성 테스트---")
        assert self.implPreProcessor.get_size() == 0
        assert self.implPreProcessor.get_raw_data() == []
        logging.info("1:---최초 생성 테스트 완료---")
        logging.info("2:---샘플 데이터 입력 테스트---")
        
        self.implPreProcessor.read_data(dataPath=self.dataPath, dataDOM=self.dataDom)
        assert self.implPreProcessor.get_size != 0
        assert self.implPreProcessor.get_raw_data() != []
        assert len(self.implPreProcessor.get_raw_data()) == self.implPreProcessor.get_size()
        
        logging.info("현재 분류된 기사 개수: " + str(self.implPreProcessor.get_size()))
        logging.info("현재 분류된 문장 개수: " + str(self.implPreProcessor.get_context_size()))
        logging.info(self.implPreProcessor.get_raw_data()[0])
        
        logging.info("2:---샘플 데이터 입력 테스트 완료---")
        
    # 클래스 소멸시 한번만 실행
    @classmethod
    def tearDownClass(self):
        pass
    
    # 각 테스트 함수 실행 시
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    @Test("데이터 정제")
    def test_remove_special_characters(self):
        test_dataset = [" test ", "<html>test</html>", "abcdef123456@naver.com test", "!t@e#$s%t^&*()", "😀😃😄t😁e😆😅s😂t", "tㅔeㅔsㅅtㅌ", "전전전전긍긍긍긍", "t   e   s   t"]
        test_answer = ["test", "test", "test", "test", "test", "test", "전전긍긍", "t e s t"]
        clean_dataset = list(map(self.implPreProcessor.remove_special_characters, test_dataset))
        self.assertEqual(clean_dataset, test_answer)
    
    @Test("마스크")
    def test_masking(self):
        sampleContext = self.implPreProcessor.get_raw_data()[0]
        tokenContext = self.implPreProcessor.tokenizer(sampleContext)
        maskContext = self.implPreProcessor.masking(tokenContext)
        # num = random.randrange(0, len(maskContext))
        ratioSum = 0.0

        for context in maskContext['input_ids']:
            num_mask = context.count(self.implPreProcessor.tokenizer.mask_token_id)
            num_pad = context.count(self.implPreProcessor.tokenizer.pad_token_id)
            ratio_mask = num_mask / (len(context) - num_pad)
            ratioSum += ratio_mask
            logging.info(f"ratio of mask_token : {ratio_mask}")
        
        assert 0.08 < ratioSum / len(maskContext['input_ids']) < 0.15, "마스킹 비율 이상. 확인 필요"

    @Test("NSP(다음 문장 예측)")
    def test_next_sentence_prediction(self):
        if self.implPreProcessor.get_size() == self.implPreProcessor.get_context_size():
            logging.warning('NSP 예측에 사용할 수 없는 데이터셋입니다.')
            return

        contextSize = self.implPreProcessor.get_context_size()
        
        #NSPmodule 기본값
        self.implPreProcessor.nsp_mode.prob = 0.5
        self.assertEqual(self.implPreProcessor.nsp_mode.prob, 0.5)

        # 잘못된 NSP 전략은 무시해야 한다
        baseStrategy = self.implPreProcessor.nsp_mode.get_strategy()
        self.assertFalse(self.implPreProcessor.nsp_mode.set_strategy("NoStrategy"))
        self.assertEqual(self.implPreProcessor.nsp_mode.get_strategy(), baseStrategy)


        testSize = contextSize // 5
        nspResult = self.implPreProcessor.next_sentence_prediction(testSize)

        # 원하는 문장의 개수만큼 해당 문서쌍을 생성해야 합니다.
        self.assertEqual(testSize, len(nspResult))
        
        # 각 데이터는 문서쌍 데이터와 해당 문제의 정답값을 가지고 있어야 합니다.
        self.assertTrue("first" in nspResult[0])
        self.assertTrue("second" in nspResult[0])
        self.assertTrue("label" in nspResult[0])

        #확률 테스트(기본값 50%)
        nextPredict = 0
        negPredict = 0
        for nspComponent in nspResult:
            if nspComponent.get('label'):
                nextPredict = nextPredict + 1
            else:
                negPredict = negPredict + 1
        logging.info("긍정 문장" + str(nextPredict) + ", 부정 문장" + str(negPredict))
        self.assertEqual(nextPredict + negPredict, testSize)
        self.assertTrue(negPredict > testSize / 10 and nextPredict > testSize / 10)

        #변동 확룔 테스트(정답 다음 문장 선택률 100%)
        self.implPreProcessor.nsp_mode.prob = 1
        nspResult = self.implPreProcessor.next_sentence_prediction(testSize)

        nextPredict = 0
        negPredict = 0
        for nspComponent in nspResult:
            if nspComponent.get('label'):
                nextPredict = nextPredict + 1
            else:
                negPredict = negPredict + 1

        self.assertEqual(nextPredict, testSize)
        self.assertEqual(negPredict, 0)

        self.implPreProcessor.nsp_mode.prob = 0

        # 다양한 문장 선택 전략 테스트
        # OnlyFirst는 오직 첫번째 문장(판별 대상 기본 문장) 기준으로 중복을 검사합니다.
        # 중복 여부 중요성이 적은 데이터를 여러번 사용하여 더 적은 데이터를 효과적으로 사용하기 위한 전략입니다.
        self.assertTrue(self.implPreProcessor.nsp_mode.set_strategy("OnlyFirst"))
        testSize = contextSize // 5
        nspResult = self.implPreProcessor.next_sentence_prediction(testSize)
        
        self.assertEqual(testSize, len(nspResult))


        # Soft는 문장 쌍 기준으로 중복 여부를 검사합니다.
        # 데이터를 사용할 수 있는 모든 쌍 대상으로 검사하여 데이터가 한정적일때 많은 nsp 데이터를 생성할 수 있습니다.
        # TODO: 함수 구조 최적화 필요
        # self.assertTrue(self.implPreProcessor.nsp_mode.setStrategy("Soft"))
        # testSize = 100
        # nspResult = self.implPreProcessor.nextSentencePrediction(testSize)
        
        # self.assertEqual(testSize, len(nspResult))


        
if __name__ == '__main__':
    main()