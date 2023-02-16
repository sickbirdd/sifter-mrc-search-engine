import os
import sys
import yaml

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from modules.lm_post_training.preprocessor import Preprocessor
from unittest import TestCase, main
from modules.utils.logging import Test, logging

with open('config_log.yaml') as f:
    """ 설정 파일 중 log 관련 설정 파일의 정보를 불러와 설정한다."""
    CONF_LOG = yaml.safe_load(f)
logging.config.dictConfig(CONF_LOG)
LOGGER = logging.getLogger('test')

class PreprocessorTest(TestCase):
    
    # 클래스 생성시 한번만 실행
    @classmethod
    def setUpClass(self):
        # 설정 파일 만들어지면 관련 변수로 대체할 것
        
        self.model_name = "klue/bert-base"
        self.impl_preprocessor  = Preprocessor(self.model_name)
        self.data_path = "datasets/lm_post_training/training/LabeledData"
        self.data_DOM = "named_entity/#/content/#/sentence".split('/')
        
        LOGGER.info("1:---최초 생성 테스트---")
        assert self.impl_preprocessor.get_size() == 0
        assert self.impl_preprocessor.get_raw_data() == []
        LOGGER.info("1:---최초 생성 테스트 완료---")
        LOGGER.info("2:---샘플 데이터 입력 테스트---")
        
        self.impl_preprocessor.read_data(data_path=self.data_path, data_DOM=self.data_DOM)
        assert self.impl_preprocessor.get_size != 0
        assert self.impl_preprocessor.get_raw_data() != []
        assert len(self.impl_preprocessor.get_raw_data()) == self.impl_preprocessor.get_size()
        
        LOGGER.info("현재 분류된 기사 개수: " + str(self.impl_preprocessor.get_size()))
        LOGGER.info("현재 분류된 문장 개수: " + str(self.impl_preprocessor.get_context_size()))
        LOGGER.info(self.impl_preprocessor.get_raw_data()[0])
        
        LOGGER.info("2:---샘플 데이터 입력 테스트 완료---")
        
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
        clean_dataset = list(map(self.impl_preprocessor.remove_special_characters, test_dataset))
        self.assertEqual(clean_dataset, test_answer)
    
    @Test("마스크")
    def test_masking(self):
        sample_context = self.impl_preprocessor.get_raw_data()[0]
        token_context = self.impl_preprocessor.tokenizer(sample_context)
        mask_context = self.impl_preprocessor.masking(token_context)
        # num = random.randrange(0, len(maskContext))
        ratio_sum = 0.0

        for context in mask_context['input_ids']:
            num_mask = context.count(self.impl_preprocessor.tokenizer.mask_token_id)
            num_pad = context.count(self.impl_preprocessor.tokenizer.pad_token_id)
            ratio_mask = num_mask / (len(context) - num_pad)
            ratio_sum += ratio_mask
            LOGGER.info(f"ratio of mask_token : {ratio_mask}")
        
        assert 0.08 < ratio_sum / len(mask_context['input_ids']) < 0.15, "마스킹 비율 이상. 확인 필요"

    @Test("NSP(다음 문장 예측)")
    def test_next_sentence_prediction(self):
        if self.impl_preprocessor.get_size() == self.impl_preprocessor.get_context_size():
            LOGGER.warning('NSP 예측에 사용할 수 없는 데이터셋입니다.')
            return

        context_size = self.impl_preprocessor.get_context_size()
        
        #NSPmodule 기본값
        self.impl_preprocessor.nsp_mode.prob = 0.5
        self.assertEqual(self.impl_preprocessor.nsp_mode.prob, 0.5)

        # 잘못된 NSP 전략은 무시해야 한다
        base_strategy = self.impl_preprocessor.nsp_mode.get_strategy()
        self.assertFalse(self.impl_preprocessor.nsp_mode.set_strategy("no_strategy"))
        self.assertEqual(self.impl_preprocessor.nsp_mode.get_strategy(), base_strategy)


        test_size = context_size // 5
        nsp_result = self.impl_preprocessor.next_sentence_prediction(test_size)

        # 원하는 문장의 개수만큼 해당 문서쌍을 생성해야 합니다.
        self.assertEqual(test_size, len(nsp_result))
        
        # 각 데이터는 문서쌍 데이터와 해당 문제의 정답값을 가지고 있어야 합니다.
        self.assertTrue("first" in nsp_result[0])
        self.assertTrue("second" in nsp_result[0])
        self.assertTrue("label" in nsp_result[0])

        #확률 테스트(기본값 50%)
        next_predict = 0
        neg_predict = 0
        for nsp_component in nsp_result:
            if nsp_component.get('label'):
                next_predict = next_predict + 1
            else:
                neg_predict = neg_predict + 1
        LOGGER.info("긍정 문장: " + str(next_predict) + ", 부정 문장: " + str(neg_predict))
        self.assertEqual(next_predict + neg_predict, test_size)
        self.assertTrue(neg_predict > test_size / 10 and next_predict > test_size / 10)

        #변동 확룔 테스트(정답 다음 문장 선택률 100%)
        self.impl_preprocessor.nsp_mode.prob = 1
        nsp_result = self.impl_preprocessor.next_sentence_prediction(test_size)

        next_predict = 0
        neg_predict = 0
        for nsp_component in nsp_result:
            if nsp_component.get('label'):
                next_predict = next_predict + 1
            else:
                neg_predict = neg_predict + 1

        self.assertEqual(next_predict, test_size)
        self.assertEqual(neg_predict, 0)

        self.impl_preprocessor.nsp_mode.prob = 0

        # 다양한 문장 선택 전략 테스트
        # OnlyFirst는 오직 첫번째 문장(판별 대상 기본 문장) 기준으로 중복을 검사합니다.
        # 중복 여부 중요성이 적은 데이터를 여러번 사용하여 더 적은 데이터를 효과적으로 사용하기 위한 전략입니다.
        self.assertTrue(self.impl_preprocessor.nsp_mode.set_strategy("only_first"))
        test_size = context_size // 5
        nsp_result = self.impl_preprocessor.next_sentence_prediction(test_size)
        
        self.assertEqual(test_size, len(nsp_result))


        # soft는 문장 쌍 기준으로 중복 여부를 검사합니다.
        # 데이터를 사용할 수 있는 모든 쌍 대상으로 검사하여 데이터가 한정적일때 많은 nsp 데이터를 생성할 수 있습니다.
        # TODO: 함수 구조 최적화 필요
        # self.assertTrue(self.impl_preprocessor .nsp_mode.set_strategy("soft"))
        # test_size  = 100
        # nsp_result = self.impl_preprocessor .next_sentence_prediction(test_size )
        
        # self.assertEqual(test_size , len(nsp_result))
        
if __name__ == '__main__':
    main()