import os
import sys
import yaml
import random

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from modules.lm_post_training.preprocessor_skeleton import PostTrainingPreprocessing as pp
from unittest import TestCase, main

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
        
        print("1:---최초 생성 테스트---")
        assert self.implPreProcessor.getSize() == 0
        assert self.implPreProcessor.getRawData() == []
        print("1:---최초 생성 테스트 완료---")
        print("2:---샘플 데이터 입력 테스트---")
        
        self.implPreProcessor.readData(dataPath=self.dataPath, dataDOM=self.dataDom)
        assert self.implPreProcessor.getSize != 0
        assert self.implPreProcessor.getRawData() != []
        assert len(self.implPreProcessor.getRawData()) == self.implPreProcessor.getSize()
        
        print("현재 분류된 기사 개수: " + str(self.implPreProcessor.getSize()))
        print("현재 분류된 문장 개수: " + str(self.implPreProcessor.getContextSize()))
        print(self.implPreProcessor.getRawData()[0])
        
        print("2:---샘플 데이터 입력 테스트 완료---")
        
    # 클래스 소멸시 한번만 실행
    @classmethod
    def tearDownClass(self):
        print("tearDownClass")
    
    # 각 테스트 함수 실행 시 
    def setUp(self):
        print('setUp')
    
    def tearDown(self):
        print('tearDown')
        
    def test_remove_special_characters(self):
        print("remove special character testing......")
        test_dataset = [" test ", "<html>test</html>", "abcdef123456@naver.com test", "!t@e#$s%t^&*()", "😀😃😄t😁e😆😅s😂t", "tㅔeㅔsㅅtㅌ", "전전전전긍긍긍긍", "t   e   s   t"]
        test_answer = ["test", "test", "test", "test", "test", "test", "전전긍긍", "t e s t"]
        clean_dataset = list(map(self.implPreProcessor.removeSpecialCharacters, test_dataset))
        self.assertEqual(clean_dataset, test_answer)
        print("Done!")
        #TODO

    def test_tokenize(self):
        print("tokenize testing......")
        # 데이터 불러오고 토크나이즈
        self.implPreProcessor.readData(dataPath=self.dataPath, dataDOM=self.dataDom)
        data_getToken = self.implPreProcessor.getTokenData()
        data_tokenize = self.implPreProcessor.tokenize(data_getToken)
        num = random.randrange(0, len(data_tokenize))
        # print(data_tokenize[num])
        # 처음과 마지막에 cls, sep 토큰 있는지 검사
        assert data_tokenize[num]["input_ids"][0] == 2, "기사 맨 앞에 cls 토큰 없음"
        for i in range(-1, -len(data_tokenize[num]["input_ids"]) - 1, -1):
            if data_tokenize[num]["input_ids"][i] != 0:
                check_last_token = (data_tokenize[num]["input_ids"][i] == 3)
                break

        assert check_last_token, "기사 맨 뒤에 sep 토큰 없음"
        # 길이가 512인지 검사
        #TODO
        assert len(data_tokenize[num]["input_ids"]) == 512, "기사 길이가 512 아님"
        print("tokenize test DONE!")
    
    def test_masking(self):
        print("masking testing......")
        self.implPreProcessor.readData(dataPath=self.dataPath, dataDOM=self.dataDom)
        data_getToken = self.implPreProcessor.getTokenData()
        data_tokenize = self.implPreProcessor.tokenize(data_getToken)
        data_masking = self.implPreProcessor.masking(data_tokenize)
        num = random.randrange(0, len(data_masking))
        # print(data_masking[0])
        num_mask = data_masking[num]["input_ids"].count(4)
        num_pad = data_masking[num]["input_ids"].count(0)
        ratio_mask = num_mask / (len(data_masking[num]["input_ids"]) - num_pad)
        print(f"ratio of mask_token : {ratio_mask}")
        assert 0.1 < ratio_mask < 0.15, "마스킹 비율 이상. 확인 필요"
        #TODO
        
    def test_masked_language_model(self):
        print("masked language model testing......")
        #TODO
        
    def test_next_sentence_prediction(self):
        print("get token data testing......")

        contextSize = self.implPreProcessor.getContextSize()
        
        #NSPmodule 기본값
        self.implPreProcessor.nspMode.prob = 0.5
        self.assertEqual(self.implPreProcessor.nspMode.prob, 0.5)

        # 잘못된 NSP 전략은 무시해야 한다
        baseStrategy = self.implPreProcessor.nspMode.getStrategy()
        self.assertFalse(self.implPreProcessor.nspMode.setStrategy("NoStrategy"))
        self.assertEqual(self.implPreProcessor.nspMode.getStrategy(), baseStrategy)


        testSize = contextSize // 5
        nspResult = self.implPreProcessor.nextSentencePrediction(testSize)

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
        self.assertEqual(nextPredict + negPredict, testSize)
        self.assertTrue(negPredict > testSize / 10 and nextPredict > testSize / 10)

        #변동 확룔 테스트(정답 다음 문장 선택률 100%)
        self.implPreProcessor.nspMode.prob = 1
        nspResult = self.implPreProcessor.nextSentencePrediction(testSize)

        nextPredict = 0
        negPredict = 0
        for nspComponent in nspResult:
            if nspComponent.get('label'):
                nextPredict = nextPredict + 1
            else:
                negPredict = negPredict + 1

        self.assertEqual(nextPredict, testSize)
        self.assertEqual(negPredict, 0)

        self.implPreProcessor.nspMode.prob = 0

        # 다양한 문장 선택 전략 테스트
        # OnlyFirst는 오직 첫번째 문장(판별 대상 기본 문장) 기준으로 중복을 검사합니다.
        # 중복 여부 중요성이 적은 데이터를 여러번 사용하여 더 적은 데이터를 효과적으로 사용하기 위한 전략입니다.
        self.assertTrue(self.implPreProcessor.nspMode.setStrategy("OnlyFirst"))
        testSize = contextSize // 5
        nspResult = self.implPreProcessor.nextSentencePrediction(testSize)
        
        self.assertEqual(testSize, len(nspResult))


        # Soft는 문장 쌍 기준으로 중복 여부를 검사합니다.
        # 데이터를 사용할 수 있는 모든 쌍 대상으로 검사하여 데이터가 한정적일때 많은 nsp 데이터를 생성할 수 있습니다.
        # TODO: 함수 구조 최적화 필요
        # self.assertTrue(self.implPreProcessor.nspMode.setStrategy("Soft"))
        # testSize = 100
        # nspResult = self.implPreProcessor.nextSentencePrediction(testSize)
        
        # self.assertEqual(testSize, len(nspResult))


        
if __name__ == '__main__':
    main()