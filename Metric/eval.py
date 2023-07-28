import time
import traceback
from multiprocessing.pool import Pool
from functools import partial
import os
import utils
from utils import TrackEvalException
import _timing
from metrics.count import Count

try:
    import tqdm
    TQDM_IMPORTED = True
except ImportError as _:
    TQDM_IMPORTED = False

# 다른 데이터셋에 대해 다양한 Metric을 평가
class Evaluator: 
    """Evaluator class for evaluating different metrics for different datasets"""

    #  평가에 대한 기본 구성 값을 포함하는 딕셔너리를 반환(config)
    @staticmethod
    def get_default_eval_config():
        """Returns the default config values for evaluation"""
        code_path = utils.get_code_path()
        default_config = {
            'USE_PARALLEL': False, # 병렬 처리를 사용할지 여부(bool)
            'NUM_PARALLEL_CORES': 8, # 병렬 처리에 사용될 코어의 개수
            'BREAK_ON_ERROR': True,  # 오류가 발생할 경우 예외를 발생시키고 프로그램을 오류와 함께 종료할지를 결정(bool)
            'RETURN_ON_ERROR': False,  # 오류 발생 시 함수에서 바로 반환할지 여부를 결정(bool)
            'LOG_ON_ERROR': os.path.join(code_path, 'error_log.txt'), #  오류를 로그 파일에 저장하는 경로를 지정

            'PRINT_RESULTS': True, # 평가 결과를 출력할지 여부(bool)
            'PRINT_ONLY_COMBINED': False, # 조합된 평가 결과만을 출력하지 여부(bool)
            'PRINT_CONFIG': True, # 기본 평가 설정 값을 출력할지 여부(bool)
            'TIME_PROGRESS': True,# 평가 과정에 시간 경과를 표시할지 여부(bool)
            'DISPLAY_LESS_PROGRESS': True, # 더 간단한 형태의 시간 경과를 표시할지 여부(bool)

            'OUTPUT_SUMMARY': True,# 요약 결과를 출력할지 여부(bool)
            'OUTPUT_EMPTY_CLASSES': True,  # 감지된 물체가 없는 클래스에 대해 요약 파일을 출력할지 여부(bool)
            'OUTPUT_DETAILED': True, #  자세한 결과를 출력할지 여부(bool)
            'PLOT_CURVES': True, # 곡선을 플롯할지 여부(bool)
        }
        return default_config
    # 초기화
    def __init__(self, config=None):
        """Initialise the evaluator with a config file"""
        self.config = utils.init_config(config, self.get_default_eval_config(), 'Eval')
        # Only run timing analysis if not run in parallel.
        if self.config['TIME_PROGRESS'] and not self.config['USE_PARALLEL']:
            _timing.DO_TIMING = True
            if self.config['DISPLAY_LESS_PROGRESS']:
                _timing.DISPLAY_LESS_PROGRESS = True

    @_timing.time
    # 주어진 데이터셋들과 메트릭들에 대해 추적 성능을 평가
    def evaluate(self, dataset_list, metrics_list, show_progressbar=False):
        """Evaluate a set of metrics on a set of datasets"""
        config = self.config # 현재 평가기의 설정을 config 변수에 저장
        metrics_list = metrics_list + [Count()]  # 평가할 메트릭들의 리스트 + count metric
        metric_names = utils.validate_metrics_list(metrics_list) # metrics_list에 있는 메트릭들의 이름들을 유효성 검사하고 metric_names 리스트에 저장
        dataset_names = [dataset.get_name() for dataset in dataset_list] # 각 데이터셋의 이름을 dataset_names 리스트에 저장
        output_res = {} # tracker별 평가 결과 저장
        output_msg = {} # tracker별 메세지 저장(success or 오류 메세지)

        for dataset, dataset_name in zip(dataset_list, dataset_names): # 데이터셋에 대해 반복 수행
            # Get dataset info about what to evaluate
            output_res[dataset_name] = {} # 평가 결과
            output_msg[dataset_name] = {} # 메세지
            tracker_list, seq_list, class_list = dataset.get_eval_info()
            print('\nEvaluating %i tracker(s) on %i sequence(s) for %i class(es) on %s dataset using the following '
                  'metrics: %s\n' % (len(tracker_list), len(seq_list), len(class_list), dataset_name,
                                     ', '.join(metric_names)))

            # Evaluate each tracker
            # tracker 모델별 반복 수행
            for tracker in tracker_list:
                # if not config['BREAK_ON_ERROR'] then go to next tracker without breaking
                try:
                    # Evaluate each sequence in parallel or in series.
                    # 각 시퀀스를 병렬 또는 순차적으로 평가
                    # returns a nested dict (res), indexed like: res[seq][class][metric_name][sub_metric field]
                    # e.g. res[seq_0001][pedestrian][hota][DetA]
                    print('\nEvaluating %s\n' % tracker)
                    time_start = time.time()
                    # 병렬 평가 사용 O
                    if config['USE_PARALLEL']: 
                        # 진행 상황을 시각적으로 보여주는 프로그레스 바를 사용할 수 있고, tqdm 모듈이 임포트되어 있는 경우에만 이 조건이 참
                        if show_progressbar and TQDM_IMPORTED:
                            seq_list_sorted = sorted(seq_list) # 평가 대상인 시퀀스 목록을 정렬하여 처리 순서를 보장
                            
                            # Pool 객체를 생성하여 병렬 평가를 수행
                            # config['NUM_PARALLEL_CORES']는 사용할 병렬 코어의 개수
                            with Pool(config['NUM_PARALLEL_CORES']) as pool, tqdm.tqdm(total=len(seq_list)) as pbar:
                                _eval_sequence = partial(eval_sequence, dataset=dataset, tracker=tracker,
                                                         class_list=class_list, metrics_list=metrics_list,
                                                         metric_names=metric_names)
                                results = []
                                # pool.imap() 메서드를 사용하여 시퀀스를 병렬로 평가
                                # chunksize는 각 작업을 처리하는 데 사용할 작업 묶음의 크기
                                for r in pool.imap(_eval_sequence, seq_list_sorted,
                                                   chunksize=20):
                                    results.append(r)
                                    pbar.update() # 프로그레스 바를 업데이트하여 진행 상황을 표시
                                res = dict(zip(seq_list_sorted, results)) # 평가 결과를 시퀀스 이름과 매핑하여 res 딕셔너리에 저장
                        # 프로그래스 바 사용 X
                        else:
                            with Pool(config['NUM_PARALLEL_CORES']) as pool:
                                _eval_sequence = partial(eval_sequence, dataset=dataset, tracker=tracker,
                                                         class_list=class_list, metrics_list=metrics_list,
                                                         metric_names=metric_names)
                                results = pool.map(_eval_sequence, seq_list)
                                res = dict(zip(seq_list, results))
                    # 병렬 평가 사용 X            
                    else:
                        res = {}
                        if show_progressbar and TQDM_IMPORTED:
                            seq_list_sorted = sorted(seq_list)
                            for curr_seq in tqdm.tqdm(seq_list_sorted):
                                res[curr_seq] = eval_sequence(curr_seq, dataset, tracker, class_list, metrics_list,
                                                              metric_names)
                        else:
                            for curr_seq in sorted(seq_list):
                                res[curr_seq] = eval_sequence(curr_seq, dataset, tracker, class_list, metrics_list,
                                                              metric_names)

                    # Combine results over all sequences and then over all classes
                    # 1. 모든 시퀀스에 대한 결과를 결합
                    # 2. 모든 클래스에 대한 결과를 결합
                    
                    # collecting combined cls keys (cls averaged, det averaged, super classes)
                    # 결합된 클래스 키들을 수집(클래스 평균, 검출 평균, 상위 클래스들)
                    
                    combined_cls_keys = [] # 결합된 클래스의 키들을 저장하는 용도로 사용(초기화) - 리스트
                    res['COMBINED_SEQ'] = {} #  모든 시퀀스에 대한 결과를 결합한 결과를 저장하는 용도로 사용(초기화) - 딕셔너리
                    # combine sequences for each class
                    # 각각의 class 별로 시퀀스들을 결합
                    for c_cls in class_list:
                        res['COMBINED_SEQ'][c_cls] = {}
                        for metric, metric_name in zip(metrics_list, metric_names):
                            curr_res = {seq_key: seq_value[c_cls][metric_name] for seq_key, seq_value in res.items() if
                                        seq_key != 'COMBINED_SEQ'}
                            res['COMBINED_SEQ'][c_cls][metric_name] = metric.combine_sequences(curr_res)
                    # combine classes
                    # class들을 결합
                    # True인 경우 클래스들을 결합
                    if dataset.should_classes_combine:
                        combined_cls_keys += ['cls_comb_cls_av', 'cls_comb_det_av', 'all']
                        res['COMBINED_SEQ']['cls_comb_cls_av'] = {}
                        res['COMBINED_SEQ']['cls_comb_det_av'] = {}
                        for metric, metric_name in zip(metrics_list, metric_names):
                            cls_res = {cls_key: cls_value[metric_name] for cls_key, cls_value in
                                       res['COMBINED_SEQ'].items() if cls_key not in combined_cls_keys}
                            res['COMBINED_SEQ']['cls_comb_cls_av'][metric_name] = \
                                metric.combine_classes_class_averaged(cls_res)
                            res['COMBINED_SEQ']['cls_comb_det_av'][metric_name] = \
                                metric.combine_classes_det_averaged(cls_res)
                    # combine classes to super classes
                    # True인 경우 상위 클래스들로 결합
                    if dataset.use_super_categories:
                        for cat, sub_cats in dataset.super_categories.items():
                            combined_cls_keys.append(cat)
                            res['COMBINED_SEQ'][cat] = {}
                            for metric, metric_name in zip(metrics_list, metric_names):
                                cat_res = {cls_key: cls_value[metric_name] for cls_key, cls_value in
                                           res['COMBINED_SEQ'].items() if cls_key in sub_cats}
                                res['COMBINED_SEQ'][cat][metric_name] = metric.combine_classes_det_averaged(cat_res)

                    # Print and output results in various formats
                    # 다양한 형식으로 결과를 출력하고 저장
                    if config['TIME_PROGRESS']:
                        print('\nAll sequences for %s finished in %.2f seconds' % (tracker, time.time() - time_start))
                    output_fol = dataset.get_output_fol(tracker) # 주어진 tracker에 대한 결과를 저장할 폴더 경로를 반환
                    tracker_display_name = dataset.get_display_name(tracker) # 주어진 tracker에 대한 이름을 반환
                    # 시퀀스 평가 결과를 담고 있는 res 변수의 키 중 'COMBINED_SEQ'를 기준으로 반복
                    # 'COMBINED_SEQ' -> 클래스 별로 결합된 평가 결과를 포함하는 딕셔너리
                    for c_cls in res['COMBINED_SEQ'].keys():  # class_list + combined classes if calculated
                        summaries = [] # 요약된 결과
                        details = [] # 자세한 결과
                        num_dets = res['COMBINED_SEQ'][c_cls]['Count']['Dets'] # 클래스 별로 결합된 평가 결과(res['COMBINED_SEQ'][c_cls])에서 'Count' 키의 'Dets' 값을 가져오기
                        if config['OUTPUT_EMPTY_CLASSES'] or num_dets > 0: # 추적된 결과가 있을 때
                            for metric, metric_name in zip(metrics_list, metric_names):
                                # for combined classes there is no per sequence evaluation
                                if c_cls in combined_cls_keys:
                                    table_res = {'COMBINED_SEQ': res['COMBINED_SEQ'][c_cls][metric_name]}
                                else:
                                    table_res = {seq_key: seq_value[c_cls][metric_name] for seq_key, seq_value
                                                 in res.items()}

                                if config['PRINT_RESULTS'] and config['PRINT_ONLY_COMBINED']:
                                    dont_print = dataset.should_classes_combine and c_cls not in combined_cls_keys
                                    if not dont_print:
                                        metric.print_table({'COMBINED_SEQ': table_res['COMBINED_SEQ']},
                                                           tracker_display_name, c_cls)
                                elif config['PRINT_RESULTS']:
                                    metric.print_table(table_res, tracker_display_name, c_cls)
                                if config['OUTPUT_SUMMARY']:
                                    summaries.append(metric.summary_results(table_res))
                                if config['OUTPUT_DETAILED']:
                                    details.append(metric.detailed_results(table_res))
                                if config['PLOT_CURVES']:
                                    metric.plot_single_tracker_results(table_res, tracker_display_name, c_cls,
                                                                       output_fol)
                            if config['OUTPUT_SUMMARY']:
                                utils.write_summary_results(summaries, c_cls, output_fol)
                            if config['OUTPUT_DETAILED']:
                                utils.write_detailed_results(details, c_cls, output_fol)

                    # Output for returning from function
                    # 결과 및 메세지(success) 저장
                    output_res[dataset_name][tracker] = res
                    output_msg[dataset_name][tracker] = 'Success'
                # 오류 처리
                except Exception as err:
                    output_res[dataset_name][tracker] = None
                    if type(err) == TrackEvalException:
                        output_msg[dataset_name][tracker] = str(err)
                    else:
                        output_msg[dataset_name][tracker] = 'Unknown error occurred.'
                    print('Tracker %s was unable to be evaluated.' % tracker)
                    print(err)
                    traceback.print_exc()
                    if config['LOG_ON_ERROR'] is not None:
                        with open(config['LOG_ON_ERROR'], 'a') as f:
                            print(dataset_name, file=f)
                            print(tracker, file=f)
                            print(traceback.format_exc(), file=f)
                            print('\n\n\n', file=f)
                    if config['BREAK_ON_ERROR']:
                        raise err
                    elif config['RETURN_ON_ERROR']:
                        return output_res, output_msg

        return output_res, output_msg


@_timing.time # 시간 측정
# 단일 시퀀스를 평가
def eval_sequence(seq, dataset, tracker, class_list, metrics_list, metric_names):
    """Function for evaluating a single sequence"""

    raw_data = dataset.get_raw_seq_data(tracker, seq) # trakcer와 시퀀스에 대한 raw data(원시 데이터)를 가져오기
    seq_res = {} # 시퀀스 평가 결과를 저장
    for cls in class_list: # class_lis -> 평가 대상 클래스들의 리스트
        seq_res[cls] = {}
        data = dataset.get_preprocessed_seq_data(raw_data, cls) # raw data를 전처리된 data로 변환(주로 입력 데이터를 특정 형식 또는 특성에 맞게 변환)
        for metric, met_name in zip(metrics_list, metric_names): # metrics_list -> 가 지표에 대한 메트릭 객체들의 리스트, metric_names -> 해당 평가 지표의 이름
            seq_res[cls][met_name] = metric.eval_sequence(data) # 시퀀스 평가 결과가 저장된 seq_res 딕셔너리를 반환
    return seq_res
