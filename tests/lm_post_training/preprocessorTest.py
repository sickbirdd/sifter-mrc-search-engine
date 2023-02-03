import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from modules.lm_post_training.preprocessor_skeleton import PostTrainingPreprocessing as pp
from unittest import TestCase, main

class preProcessorTest(TestCase):
    
    # 클래스 생성시 한번만 실행
    @classmethod
    def setUpClass(self):
        # 설정 파일 만들어지면 관련 변수로 대체할 것
        self.implPreProcessor = pp(modelName="klue/bert-base")
        self.labelDataPath = "" 
        self.sourceDataPath = ""
        
    # 클래스 소멸시 한번만 실행
    @classmethod
    def tearDownClass(self):
        print("tearDownClass")
    
    # 각 테스트 함수 실행 시 
    def setUp(self):
        print('setUp')
    
    def tearDown(self):
        print('tearDown')
        
    def test_get_token_data(self):
        print("get token data testing......")
        #TODO
        
    def test_context_finder(self):
        print("context finder testing......")
        #TODO
        
    def test_remove_special_characters(self):
        print("remove special character testing......")
        #TODO
        
    def test_masking(self):
        print("masking testing......")        
        #TODO
    
    def test_tokenize(self):
        print("tokenize testing......")
        #TODO
        
    def test_masked_language_model(self):
        print("masked language model testing......")
        #TODO
        
    def test_next_sentence_prediction(self):
        print("get token data testing......")
        #NSPmodule 기본값
        self.implPreProcessor.nspMode.prob = 0.5
        self.assertEqual(self.implPreProcessor.nspMode.prob, 0.5)
        self.assertEqual(self.implPreProcessor.nspMode.setStrategy("NoStrategy"))


        testSize = 1000
        nspResult = self.implPreProcessor.nextSentencePrediction(testSize)

        # 원하는 문장의 개수만큼 해당 문서쌍을 생성해야 합니다.
        self.assertEqual(testSize, len(nspResult))
        
        # 각 데이터는 문서쌍 데이터와 해당 문제의 정답값을 가지고 있어야 합니다.
        self.assertTrue("data" in nspResult[0])
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
        self.assertTrue(negPredict < testSize / 10)


        
if __name__ == '__main__':
    main()