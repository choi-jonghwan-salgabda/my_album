"""
SimpleLogger.py

이 모듈은 콘솔 출력 및 파일 기록(동기/비동기)을 지원하는 간단한 로깅 클래스인 SimpleLogger를 제공합니다.
또한, 로그 레벨 관리, 호출 함수 이름 포함, 딕셔너리/리스트의 보기 좋은 출력(pretty print) 등의 기능을 포함합니다.
명령줄 인자 파싱 유틸리티 함수(get_argument)와 숫자 자릿수 계산 함수(calc_digit_number),
그리고 시각적 문자열 길이 계산 함수(visual_length)도 함께 제공됩니다.

주요 구성 요소:
- LOG_LEVELS: 로그 레벨 이름과 해당 정수 값을 매핑하는 딕셔너리.
- SimpleLogger 클래스: 핵심 로깅 기능을 제공.
- 유틸리티 함수: reset_logger, calc_digit_number, get_argument, visual_length.
- logger: SimpleLogger의 공유 인스턴스로, 다른 모듈에서 import하여 사용할 수 있습니다.
"""

import datetime
import os
import pprint
import inspect
import queue # queue 모듈 임포트
import threading # threading 모듈 임포트
import time # 필요에 따라 스레드 대기 등에 사용될 수 있습니다.
import yaml
import math
import argparse
from pathlib import Path
from datetime import datetime
import traceback
import unicodedata # visual_length 함수에서 사용

# 로그 레벨 이름과 해당 정수 값을 정의합니다. 숫자가 낮을수록 더 상세한 로그 레벨입니다.
LOG_LEVELS = {
    "DEBUG": 10,    # 디버깅 목적으로 사용되는 상세 정보
    "INFO": 20,     # 일반 정보성 메시지
    "WARNING": 30,  # 경고 메시지 (잠재적 문제)
    "ERROR": 40,    # 오류 메시지 (기능 수행 불가)
    "CRITICAL": 50  # 심각한 오류 메시지 (애플리케이션 중단 가능)
}

# 비동기 파일 쓰기 스레드를 안전하게 종료시키기 위한 특별한 객체(Sentinel)입니다.
_SENTINEL = object()

def make_normaized_path(arg_path:Path)->Path:
    # Configger 클래스 또는 경로 처리 함수 내에서
    # resolved_path = resolve_placeholders(raw_path_from_yaml) # 기존 로직

    # 1. 사용자 홈 디렉토리 확장
    expanded_path = os.path.expanduser(arg_path)

    # 2. 경로 정규화 (중복 슬래시 제거 등)
    normalized_path = os.path.normpath(expanded_path)

    # 최종적으로 normalized_path 사용
    return normalized_path

class SimpleLogger:
    """
    콘솔 출력 및 파일 기록(동기/비동기)을 지원하는 간단한 로깅 클래스입니다.
    로그 레벨, 호출 함수 이름 포함 여부, 딕셔너리/리스트의 보기 좋은 출력(pretty print) 등을 설정할 수 있습니다.
    """
    def __init__(self, min_level="INFO", pretty_print=True, standalone=False):
        # 로거 초기화
        self.LOG_LEVELS = LOG_LEVELS
        self._logger_path = None
        self._min_level = self._validate_level(min_level)
        self._min_level_int = self.LOG_LEVELS[self._min_level]
        self._include_function_name = False
        self._pretty_print = pretty_print

        self._async_file_writing_enabled = False # 비동기 파일 쓰기 활성화 여부
        self._log_queue = None                   # 비동기 로깅 시 사용할 메시지 큐
        self._stop_event = None                  # 비동기 쓰기 스레드 종료 이벤트
        self._writer_thread = None               # 비동기 쓰기 스레드 객체
        self._file_handle = None                 # 파일 핸들 (비동기 쓰기 스레드에서 사용)
        self._initialized = False                # 로거가 독립 실행 모드로 초기화되었는지 여부

        if standalone:
            # 독립 실행 모드일 경우, 자체 설정 파일을 로드하여 로거를 설정합니다.
            self._apply_standalone_config()

    def _validate_level(self, level_str):
        """
        입력된 로그 레벨 문자열을 검증하고, 유효하면 대문자로 변환하여 반환합니다.
        유효하지 않으면 현재 설정된 최소 로그 레벨을 유지합니다.

        Args:
            level_str (str): 검증할 로그 레벨 문자열 (예: "INFO", "debug").

        Returns:
            str: 유효한 대문자 로그 레벨 문자열.
        """
        if level_str is None:
            return self._min_level
        upper_level = str(level_str).upper()
        if upper_level in self.LOG_LEVELS:
            # 입력 레벨이 None이면 현재 레벨 유지
            return upper_level
        print(f"경고: 유효하지 않은 로그 레벨 '{level_str}'입니다. 현재 설정 '{self._min_level}'을 유지합니다.")
        return self._min_level

    def _apply_standalone_config(self):
        current_script_path = os.path.realpath(__file__)
        current_directory = os.path.dirname(current_script_path)

        """
        독립 실행 모드일 때 로거 설정을 적용합니다.
        'logging.yaml' 파일이 현재 스크립트 디렉토리에 있으면 해당 설정을 로드하고,
        없으면 기본 설정을 사용합니다.
        """
        # 기본 로그 디렉토리 및 레벨 설정
        log_dir = Path(current_directory).parent.parent
        log_level = self._min_level  # ✅ 올바르게 참조
        log_format = "%(asctime)s - %(levelname)-6s - %(funcName)-20s (configger_standalone) - %(message)s"

        # 로깅 설정 파일 경로
        cfg_yaml_file_name = "logging.yaml"
        cfg_path = os.path.join(current_directory, cfg_yaml_file_name)

        if os.path.exists(cfg_path):
            # 설정 파일이 존재하면 로드
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    config_yaml = yaml.safe_load(f) or {}
                log_cfg = config_yaml.get("logging", {})

                # 파일에서 설정 값 가져오기 (없으면 기본값 사용)
                log_dir = log_cfg.get("log_dir", log_dir)
                self._min_level = log_cfg.get("level", log_level)
                log_format = log_cfg.get("log_format", log_format)
            except Exception as e:
                print(f"오류: configger용 로깅 구성 파일 '{cfg_path}' 읽기 실패 - {e}")

        # 로그 디렉토리가 없으면 생성
        if  not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # 로그 파일 경로 설정
        script_name = os.path.splitext(os.path.basename(current_script_path))[0]
        date_str = datetime.now().strftime("%y%m%d")
        log_path = os.path.join(log_dir, f'{script_name}_{date_str}.log')

        # 로거 설정 적용
        self.setup(
            logger_path=log_path,
            min_level=self._min_level,
            include_function_name=True,
            pretty_print=True,
            async_file_writing=False
        )
        print(f"독립 실행용 로깅 설정 완료: {log_path}")
        self._initialized = True # 독립 실행 설정 완료 플래그

    def setup(self, logger_path=None, min_level=None, include_function_name=None, pretty_print=None, async_file_writing=None):
        """
        로거 설정을 변경합니다. 각 인자는 None이 아닌 경우에만 해당 설정을 업데이트합니다.

        Args:
            logger_path (str, optional): 로그를 기록할 파일 경로. None이면 파일 기록 비활성화.
            min_level (str, optional): 기록할 최소 로그 레벨.
            include_function_name (bool, optional): 로그 메시지에 호출 함수 이름을 포함할지 여부.
            pretty_print (bool, optional): 딕셔너리/리스트를 예쁘게 출력할지 여부.
            async_file_writing (bool, optional): 비동기 파일 기록 사용 여부.
        """
        # 변경될 수 있는 새로운 경로 임시 저장 (async 설정에 영향 줄 수 있음)
        new_logger_path = logger_path if logger_path is not None else self._logger_path

        # 변경될 수 있는 비동기 설정 임시 저장
        desired_async_state = async_file_writing if async_file_writing is not None else self._async_file_writing_enabled

        # --- 비동기 쓰기 스레드 상태 변화 감지 및 처리 로직 ---
        # Case 1: 비동기 쓰기 비활성화 요청 또는 경로가 None이 됨 (async 중지)
        # 또는 비동기 활성화 상태에서 경로가 변경됨 (async 중지 후 재시작)
        is_path_changing = (logger_path is not None and logger_path != self._logger_path)
        needs_async_stop = (
            self._async_file_writing_enabled and
            (desired_async_state is False or new_logger_path is None or is_path_changing)
        )

        if needs_async_stop:
            # 비동기 쓰기 스레드 중지
            self._stop_async_writer()

        # Case 2: 비동기 쓰기 활성화 요청 또는 비동기 활성화 상태에서 경로가 변경됨 (async 시작/재시작)
        # 단, 새 경로가 유효해야 함
        needs_async_start = (
             desired_async_state is True and
             new_logger_path is not None and # 유효한 경로 필요
             (not self._async_file_writing_enabled or is_path_changing) # 현재 비활성화 상태이거나 경로가 변경된 경우
        )

        # 로거 경로 업데이트 (stop 로직 이후, start 로직 이전에)
        if logger_path is not None:
             # 새 로그 파일 경로 설정
             self._logger_path = logger_path
             # 만약 async_file_writing 설정은 None으로 넘어왔는데 logger_path가 None이 되었고 async가 켜져있었다면
             # 위 needs_async_stop에서 이미 async는 꺼졌을 것입니다.

        if needs_async_start:
             # 새 경로로 비동기 쓰기 스레드 시작
             self._start_async_writer(self._logger_path)

        # --- 나머지 설정 업데이트 ---
        if min_level is not None:
            # 최소 로그 레벨 업데이트
            self._min_level = self._validate_level(min_level)
            self._min_level_int = self.LOG_LEVELS[self._min_level]
        if include_function_name is not None:
            # 함수 이름 포함 여부 업데이트
            self._include_function_name = bool(include_function_name)
        if pretty_print is not None:
            # Pretty print 사용 여부 업데이트
             self._pretty_print = bool(pretty_print)
        # self._async_file_writing_enabled는 _start_async_writer/_stop_async_writer 에서 설정됨

        # 비동기 쓰기가 요청되었으나 경로가 없을 경우 경고
        if async_file_writing is True and self._logger_path is None:
            print("경고: 비동기 파일 기록을 요청했지만 로그 파일 경로가 지정되지 않았습니다. 비동기 기록을 활성화할 수 없습니다.")

    def _async_writer_task(self, log_file_path):
        """
        비동기 파일 쓰기 스레드의 메인 작업 함수입니다.
        로그 메시지 큐에서 메시지를 가져와 지정된 로그 파일에 기록합니다.
        _SENTINEL 객체를 받으면 스레드를 종료합니다.

        Args:
            log_file_path (str): 로그를 기록할 파일 경로.
        """
        print(f"비동기 로깅 스레드 시작. 파일: {log_file_path}")
        try:
            # 디렉토리가 없으면 생성
            log_dir = os.path.dirname(log_file_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            # 파일 열기
            # Note: 파일 핸들을 스레드 내부에 유지합니다.
            with open(log_file_path, 'a', encoding='utf-8') as f: # 추가 모드('a')로 파일 열기
                self._file_handle = f # 클래스 속성에 파일 핸들 저장 (shutdown에서 필요할 수 있음)
                while not self._stop_event.is_set():
                    try:
                        # 큐에서 메시지 가져오기 (짧은 타임아웃 사용)
                        # 타임아웃이 없으면 stop_event.is_set()을 확인하기 어려움
                        message = self._log_queue.get(timeout=0.1)

                        if message is _SENTINEL:
                            # 종료 Sentinel을 받으면 루프 종료
                            print("비동기 로깅 스레드: 종료 Sentinel 수신.")
                            break
                        
                        # 메시지 파일에 쓰기
                        f.write(message + '\n')
                        f.flush() # 매 메시지마다 파일 버퍼를 비워 디스크에 즉시 기록 (안전하지만 성능에 영향 가능)
                        self._log_queue.task_done() # 큐 작업 완료 알림

                    except queue.Empty:
                        # 큐가 비어있으면 타임아웃 발생. 루프 조건(stop_event) 다시 확인.
                        pass
                    except Exception as e:
                        # 파일 쓰기 중 오류 발생 시 콘솔에 알림
                        print(f"오류: 비동기 쓰기 스레드에서 파일 '{log_file_path}'에 쓰기 실패: {e}")
                        # 심각한 오류 시 스레드를 중지할 수도 있지만, 여기서는 계속 시도
                        if self._log_queue.qsize() > 0: # 오류 발생했더라도 메시지는 큐에서 제거
                             self._log_queue.task_done()

        except Exception as e:
            print(f"오류: 비동기 로깅 스레드 초기화 또는 파일 열기 실패: {e}")

        finally:
            # 루프 종료 후 파일 닫기
            if self._file_handle and not self._file_handle.closed:
                 self._file_handle.close()
                 print(f"비동기 로깅 스레드: 파일 '{log_file_path}' 닫힘.")
            self._file_handle = None # 핸들 해제
            print("비동기 로깅 스레드 종료.")

    def _start_async_writer(self, path):
        """
        비동기 파일 쓰기 스레드를 시작합니다.
        이미 실행 중이거나 로그 파일 경로가 없으면 시작하지 않습니다.

        Args:
            path (str): 로그를 기록할 파일 경로.
        """
        if self._async_file_writing_enabled:
             # 이미 스레드가 실행 중이면 경고 출력 후 반환
             print("경고: 비동기 쓰기 스레드가 이미 실행 중입니다.")
             return

        if path is None:
             # 로그 파일 경로가 없으면 경고 출력 후 반환
             print("경고: 비동기 쓰기 활성화를 위해 유효한 로그 파일 경로가 필요합니다.")
             return

        print(f"비동기 쓰기 스레드 시작 요청. 경로: {path}")
        self._logger_path = path # 스레드 시작 전에 경로 확정
        self._log_queue = queue.Queue()      # 로그 메시지를 담을 큐 생성
        self._stop_event = threading.Event() # 스레드 종료를 위한 이벤트 객체 생성
        # 데몬 스레드 사용 시 메인 프로그램 종료 시 강제 종료될 수 있음
        # 명시적인 shutdown() 호출을 권장합니다.
        self._writer_thread = threading.Thread(target=self._async_writer_task, args=(self._logger_path,), daemon=False) # daemon=False로 명시적 종료 대기
        self._writer_thread.start()
        self._async_file_writing_enabled = True
        print("비동기 쓰기 스레드 시작 완료.")

    def _stop_async_writer(self):
        """
        실행 중인 비동기 파일 쓰기 스레드를 안전하게 종료합니다.
        큐에 _SENTINEL을 넣어 스레드에 종료를 알리고, 스레드가 완료될 때까지 대기합니다.
        """
        if not self._async_file_writing_enabled:
            # 스레드가 실행 중이 아니면 정보 메시지 출력 후 반환 (또는 아무것도 안 함)
            # print("정보: 비동기 쓰기 스레드가 실행 중이 아닙니다.")
            return

        print("비동기 쓰기 스레드 종료 요청 중...")
        if self._log_queue:
             # 큐에 종료 Sentinel을 넣어 쓰기 스레드가 이를 처리하고 종료하도록 유도
             self._log_queue.put(_SENTINEL) # 종료 Sentinel을 큐에 넣음
        if self._stop_event:
             # 종료 이벤트를 설정하여 스레드가 루프를 빠져나오도록 함
             self._stop_event.set() # 종료 이벤트를 설정

        if self._writer_thread and self._writer_thread.is_alive():
            # 스레드가 종료될 때까지 최대 5초간 대기
            self._writer_thread.join(timeout=5) # 스레드 종료 대기 (최대 5초)
            if self._writer_thread.is_alive():
                 print("경고: 비동기 쓰기 스레드가 시간 내에 종료되지 않았습니다.")
        else:
             print("경고: 종료할 비동기 쓰기 스레드가 없거나 이미 종료되었습니다.")

        # 사용된 리소스 정리
        self._log_queue = None # 큐 객체 해제
        self._stop_event = None
        self._writer_thread = None
        # self._file_handle은 쓰레드 내부에서 닫힙니다.
        self._async_file_writing_enabled = False
        print("비동기 쓰기 스레드 종료 처리 완료.")

    def shutdown(self):
        """
        로거를 종료합니다. 실행 중인 비동기 파일 쓰기 스레드가 있다면 안전하게 종료시킵니다.
        애플리케이션 종료 시 호출하는 것이 좋습니다.
        """
        self._stop_async_writer()
        print("로거 종료 완료.")

    def __del__(self):
        """
        SimpleLogger 객체가 소멸될 때 호출됩니다.
        실행 중인 비동기 쓰기 스레드가 있다면 종료를 시도합니다 (최선의 노력, 항상 보장되지는 않음).
        명시적으로 shutdown()을 호출하는 것이 더 안전합니다.
        """
        # print("정보: SimpleLogger 객체 소멸자 호출됨.")
        self.shutdown()

    def get_config(self):
        """
        현재 로거의 주요 설정 값들을 딕셔너리 형태로 반환합니다.
        """
        return {
            "logger_path": self._logger_path, # 로그 파일 경로
            "min_level": self._min_level,
            "min_level_int": self._min_level_int,
            "include_function_name": self._include_function_name,
            "pretty_print": self._pretty_print,
            "async_file_writing_enabled": self._async_file_writing_enabled,
            "async_queue_size": self._log_queue.qsize() if self._log_queue else 0, # 현재 큐 크기
            "async_thread_alive": self._writer_thread.is_alive() if self._writer_thread else False # 쓰레드 활성화 상태
        }

    # 레벨 문자열을 받아서 해당 정수 값을 반환하는 메소드
    def get_level_value(self, level_str):
        """
        주어진 로그 레벨 문자열(예: "INFO")에 해당하는 정수 값을 반환합니다.

        Args:
            level_str (str): 로그 레벨 문자열.

        Returns:
            int: 해당 로그 레벨의 정수 값. 유효하지 않으면 INFO 레벨의 정수 값을 반환.
        """
        upper_level = str(level_str).upper()
        # 내부 맵 또는 클래스 속성 상수를 사용하여 값 반환
        # return self._LOG_LEVEL_MAP.get(upper_level, self._LOG_LEVEL_MAP["INFO"]) # 맵 사용 시
        return getattr(self, upper_level, self.INFO) # 클래스 속성 및 getattr 사용 시

    def show_config(self):
        """현재 로거 설정을 콘솔에 보기 좋게(pretty print) 출력합니다."""
        print("\n--- 현재 로거 설정 ---")
        pprint.pprint(self.get_config(), indent=4, width=80)
        print("----------------------")

    def _get_caller_function_name(self):
        """로깅 함수를 호출한 상위 함수 이름을 가져옵니다."""
        if not self._include_function_name:
            # 함수 이름 포함 설정이 꺼져있으면 None 반환
            return None

        # 현재 실행 중인 프레임 정보를 가져옵니다.
        # inspect.currentframe()은 현재 함수의 프레임을 반환합니다.
        # 호출 스택을 거슬러 올라가 로깅 함수를 호출한 함수의 이름을 찾습니다.
        frame = inspect.currentframe()
        # 스택 구조: currentframe -> _get_caller_function_name -> log -> level_method -> user_function
        # user_function의 프레임은 currentframe 기준으로 4단계 위에 있습니다.
        # frame0: currentframe (_get_caller_function_name 내부)
        # frame1: _format_message 또는 log 호출
        # frame2: log 또는 level_method 호출
        # frame3: level_method 또는 user_function 호출
        # frame4: user_function (목표)

        caller_frame = None
        try:
            # 4단계 위 프레임을 얻어보자. (simple_logger 클래스 내 메소드 호출 스택 고려)
            # log -> info/debug/etc -> user_function
            # _format_message -> log -> info/debug/etc -> user_function
            # _get_caller_function_name -> _format_message -> log -> info/debug/etc -> user_function
            # 즉 _get_caller_function_name에서 user_function까지는 4단계
            # 스택 프레임이 충분히 깊은지 확인하여 AttributeError 방지
            if frame and frame.f_back and frame.f_back.f_back and frame.f_back.f_back.f_back and frame.f_back.f_back.f_back.f_back:
                 caller_frame = frame.f_back.f_back.f_back.f_back
                 function_name = caller_frame.f_code.co_name
                 return function_name
            elif frame and frame.f_back and frame.f_back.f_code.co_name == 'log': # 직접 log()를 호출한 경우
                 caller_frame = frame.f_back
                 function_name = caller_frame.f_code.co_name
                 return function_name
            else:
                 return "UnknownFunction" # 스택 깊이가 예상과 다르거나 프레임이 없는 경우

        except Exception:
            return "ErrorGettingFunctionName"
        finally:
            # 프레임 객체는 순환 참조를 유발할 수 있으므로 사용 후 명시적으로 해제하는 것이 좋습니다.
            del frame
            # caller_frame도 필요 이상 유지하지 않도록 처리 (여기서는 반환 후 GC 대상)

    def _format_message(self, message, level):
        """
        주어진 메시지와 로그 레벨을 사용하여 최종 로그 메시지 문자열을 생성합니다.
        타임스탬프, 로그 레벨, (설정 시) 함수 이름을 포함하며,
        메시지 내용이 딕셔너리나 리스트인 경우 pretty print를 적용합니다.

        Args:
            message (any): 로깅할 메시지 내용.
            level (str): 메시지의 로그 레벨 문자열 (예: "INFO").

        Returns:
            str: 포맷팅된 최종 로그 메시지 문자열.
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        pid = os.getpid() # 현재 프로세스 ID 가져오기
        header_parts = [f"[{pid:>5}]", f"[{timestamp}]", f"[{(level.upper()):8s}]"] # PID 추가 (5자리로 오른쪽 정렬)

        # 함수 이름 포함 설정이 켜져있으면 함수 이름을 가져와 헤더에 추가
        if self._include_function_name:
            # 함수 이름 포함 설정이 켜져있으면, 호출 함수 이름을 가져와 헤더에 추가
            function_name = self._get_caller_function_name()
            if function_name and function_name != '<module>': # 함수 이름을 성공적으로 가져왔고 메인 모듈이 아니면 추가
                header_parts.append(f"[{function_name}]")

        header = " ".join(header_parts)

        # 메시지 내용 포맷팅 (pretty_print 설정에 따름)
        if isinstance(message, (dict, list)) and self._pretty_print:
            # 메시지가 딕셔너리 또는 리스트이고 pretty_print가 활성화된 경우,
            # pprint.pformat을 사용하여 여러 줄로 보기 좋게 포맷합니다.
            message_content = pprint.pformat(message, indent=4, width=80, compact=False)
            formatted_output = f"{header}\n{message_content}"
        else:
            # 그 외 타입은 문자열로 변환
            message_content = str(message)
            formatted_output = f"{header} {message_content}" # 헤더와 한 줄로 표시

        return formatted_output

    def _write_to_file(self, formatted_message):
        """
        포맷된 로그 메시지를 파일에 기록합니다.
        비동기 파일 쓰기가 활성화된 경우 큐에 메시지를 넣고,
        그렇지 않으면 동기적으로 파일에 직접 씁니다.

        Args:
            formatted_message (str): 파일에 기록할, 이미 포맷팅된 로그 메시지.
        """
        if not self._logger_path:
            return # 로그 파일 경로가 설정되지 않았으면 아무 작업도 하지 않음

        if self._async_file_writing_enabled:
            # 비동기 모드: 메시지를 큐에 넣음
            if self._log_queue:
                try:
                    self._log_queue.put(formatted_message) # 큐에 메시지 추가
                except Exception as e:
                    print(f"오류: 비동기 큐에 메시지 추가 실패: {e}")
        else:
            # 동기 모드: 즉시 파일에 쓰기
            try:
                # 디렉토리가 없으면 생성
                log_dir = os.path.dirname(self._logger_path)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)

                # 파일에 메시지 추가 모드로 쓰기
                with open(self._logger_path, 'a', encoding='utf-8') as f: # 'a' (append) 모드로 파일 열기
                    f.write(formatted_message + '\n')
            except Exception as e:
                # 파일 쓰기 실패 시 에러 메시지 출력 (기본 print 사용)
                print(f"오류: 동기 모드에서 로그 파일 '{self._logger_path}'에 쓰기 실패: {e}")

    def log(self, message, level="INFO"):
        """
        핵심 로깅 메소드. 최소 레벨 검사 후 메시지를 포맷하고 출력합니다.

        Args:
            message: 출력할 메시지.
            level (str): 메시지의 로그 레벨.
        """
        level_str = self._validate_level(level) # 입력된 레벨 문자열 검증
        level_int = self.LOG_LEVELS[level_str]  # 검증된 레벨의 정수 값 가져오기

        # 설정된 최소 레벨보다 낮은 레벨의 메시지는 무시
        if level_int < self._min_level_int: # 현재 메시지 레벨이 설정된 최소 레벨보다 낮으면 기록하지 않음
            return

        # 메시지 포맷팅 (함수 이름 포함 및 pretty_print 적용)
        formatted_message = self._format_message(message, level_str)

        # 콘솔에 출력
        print(formatted_message)

        # 파일 경로가 지정된 경우 파일에 기록 (동기/비동기는 내부에서 결정)
        self._write_to_file(formatted_message)

    def error(self, message, exc_info=False): # exc_info 인자 추가
        """
        ERROR 레벨의 로그를 기록합니다.
        exc_info=True로 설정하면 현재 예외 정보(트레이스백)를 함께 기록합니다.

        Args:
            message (any): 로깅할 메시지.
            exc_info (bool, optional): True이면 현재 예외 정보를 로그에 포함. 기본값은 False.
        """
        # exc_info가 True이면 현재 예외의 트레이스백 정보를 메시지에 추가
        log_message = message
        if exc_info:
            # sys.exc_info()는 현재 처리 중인 예외 정보를 반환합니다.
            # traceback.format_exc()는 현재 예외의 전체 트레이스백을 문자열로 반환합니다.
            # 호출하는 쪽에서 try-except 블록 내에서 호출될 때 유용합니다.
            # 만약 예외 객체를 직접 전달받는다면 다른 방식을 사용할 수 있습니다.
            exc_text = traceback.format_exc()
            log_message = f"{message}\n{exc_text}"
        self.log(log_message, level="ERROR")

    # --- 사용자 편의를 위한 레벨별 로깅 메소드들 ---
    def debug(self, message):
        """DEBUG 레벨의 로그를 기록합니다."""
        self.log(message, level="DEBUG")

    def info(self, message):
        """INFO 레벨의 로그를 기록합니다."""
        self.log(message, level="INFO")

    def warning(self, message):
        """WARNING 레벨의 로그를 기록합니다."""
        self.log(message, level="WARNING")

    def critical(self, message):
        """CRITICAL 레벨의 로그를 기록합니다."""
        self.log(message, level="CRITICAL")

def reset_logger(min_level = 'INFO'):
    """
    애플리케이션 전체 로깅 설정을 초기화하거나 재설정합니다.
    이 함수는 공유 `logger` 인스턴스의 설정을 변경합니다.
    """
    # 로그 파일 경로, 레벨 등을 결정합니다.
    # 예를 들어, 애플리케이션 루트 디렉토리 아래 logs 폴더에 저장
    app_root = Path(__file__).resolve().parent # my_yolo_tiny.py가 있는 디렉토리
    log_directory = app_root / "logs"
    log_file_name = f"{app_root.name}_{datetime.now().strftime('%Y%m%d')}.log"
    log_file_path = log_directory / log_file_name

    logger.setup(
        logger_path=str(log_file_path), # 로그 파일 경로 설정
        min_level=min_level,  # 애플리케이션 기본 로그 레벨
        include_function_name=True,
        pretty_print=True,
        async_file_writing=False # 필요에 따라 True로 설정
    )
    logger.info(f"애플리케이션 로거 초기화 완료. 로그 파일: {log_file_path}")
    logger.show_config()

def calc_digit_number(in_number: int) -> int:
    """
    주어진 정수의 자릿수를 계산하여 반환합니다.

    Args:
        in_number (int): 자릿수를 계산할 정수.

    Returns:
        int: 정수의 자릿수. (예: 0 -> 1, 123 -> 3, -12 -> 2)
    """
    # 0은 특별히 한 자리 숫자입니다.
    if in_number == 0:
        return 1
    # 음수의 경우 절댓값을 사용합니다.
    # elif in_number < 0: # abs()를 사용하므로 이 조건은 불필요합니다.
        in_number = abs(in_number)

    # 양의 정수 N의 자릿수는 floor(log10(N)) + 1 공식을 사용합니다.
    # math.log10(N)은 N의 상용로그 값을 계산합니다.
    # math.floor()는 소수점 이하를 버립니다.
    return math.floor(math.log10(in_number)) + 1

def visual_length(text, space_width=1):
    """
    주어진 텍스트의 시각적 길이를 계산합니다.
    일반 문자는 1, 전각 문자(한글, 한자 등)는 2, 공백은 space_width로 계산합니다.

    Args:
        text (str): 길이를 계산할 텍스트.
        space_width (int, optional): 공백 문자의 시각적 너비. 기본값은 1.

    Returns:
        int: 텍스트의 시각적 길이.
    """
    length = 0
    for ch in text:
        if ch == ' ':
            length += space_width # 공백 처리
        elif unicodedata.east_asian_width(ch) in ('W', 'F'):
            length += 2 # 전각 문자(Wide, Fullwidth)는 2로 계산
        else:
            length += 1 # 그 외 문자(반각 등)는 1로 계산
    return length



def get_argument() -> argparse.Namespace:
    """
    명령줄 인자를 파싱하여 스크립트 실행에 필요한 경로들을 설정합니다.
    - 루트 디렉토리 (`--root-dir`)
    - 로그 파일 저장 디렉토리 (`--log-dir`)
    - 설정 파일 경로 (`--config-path`)

    인자가 제공되지 않으면, `--root-dir`를 기준으로 기본값을 동적으로 계산하여 사용합니다.

    Returns:
        argparse.Namespace: 파싱된 명령줄 인자들을 담고 있는 객체.
                            이 객체는 `root_dir`, `log_dir`, `config_path` 속성을 가집니다.
    """

    # argparse.ArgumentParser 객체 생성 및 설명 추가
    parser = argparse.ArgumentParser(description="스크립트 실행을 위한 경로 및 로깅 레벨 설정")
    parser.add_argument(
        '--root-dir', '-root',
        type=str,
        default=os.getcwd(),
        help='프로젝트의 루트 디렉토리. (기본값: 현재 작업 디렉토리)'
    )
    parser.add_argument(
        '--log-dir', '-log',
        type=str,
        default=True, # 초기값을 None으로 설정하고 아래에서 동적으로 할당
        help='로그 파일을 저장할 디렉토리. (꼭 입력해야함)' # 로그 파일이 저장될 디렉토리
    )
    parser.add_argument(
        '--log-level', '-lvl',
        type=str,
        default='warning',
        choices=["debug", "info", "warning", "error", "critical"],
        help='로그 Level을 지정하는 값. (기본값: warning)' # 기록할 로그의 최소 레벨
    )
    parser.add_argument(
        '--config-path', '-cfg',
        type=str,
        default=True, # 초기값을 None으로 설정하고 아래에서 동적으로 할당
        help='설정 파일(YAML)의 경로. (꼭 입력해야함)' # YAML 설정 파일 경로
    )
    parser.add_argument(
        '--source-dir', '-src',
        type=str,
        required=False,
        help='원천(source) 디렉토리. (기본값: 제공되지 않음)' # 소스 파일들이 있는 디렉토리
    )
    parser.add_argument(
        '--destination-dir', '-dst',
        type=str,
        required=False,
        help='결과물(destination) 디렉토리. (기본값: 제공되지 않음)' # 처리 결과가 저장될 디렉토리
    )
    parser.add_argument(
        '--target-dir', '-tgt',
        type=str,
        required=False,
        help='작업대상(target) 디렉토리. (기본값: 제공되지 않음)' # 특정 작업의 대상이 되는 디렉토리
    )

    parser.add_argument(
        '--dry_run', '-dry',
        action="store_true",
        help="실제 파일 이동/삭제 없이 로그만 출력합니다." # 특정 작업의 대상이 되는 디렉토리
    )

    # 명령줄 인자 파싱
    args = parser.parse_args()

    # # --log-dir 인자가 제공되지 않았으면 기본값 설정
    # if args.log_dir is None:
    #     args.log_dir = os.path.join(args.root_dir, 'logs')
    #     print(f"정보: --log-dir      인자가 제공되지 않았습니다. 기본 로그 디렉토리 '{args.log_dir}'을 사용합니다.")
    
    # 로그 디렉토리 생성 (존재하지 않으면)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    # # --config-path 인자가 제공되지 않았으면 기본값 설정
    # # 프로젝트 이름을 기반으로 설정 파일 경로를 동적으로 생성합니다.
    # if args.config_path is None:
    #     config_path = (Path(args.root_dir) / '../config' / f"photo_album.yaml").expanduser().resolve()
    #     args.config_path = str(config_path)

    #     # 정보 메시지 형식 변경
    #     print(f"정보: --config-path 인자가 제공되지 않았습니다. 기본 설정 파일 경로 '{args.config_path}'을 사용합니다.")

    # 파싱된 인자 정보 출력
    # --- 파싱된 인자 정보 출력 (visual_length 적용) ---
    arg_print_definitions = [
        ("루트 디렉토리 (--root-dir)", lambda: args.root_dir, lambda: True),
        ("로그 디렉토리 (--log-dir)", lambda: args.log_dir, lambda: True),
        ("로그 레벨설정 (--log-level)", lambda: args.log_level, lambda: True), # 레이블 수정
        ("설정 파일경로 (--config-path)", lambda: args.config_path, lambda: True),
    ]
    if args.source_dir is not None:
        arg_print_definitions.append(("기준 디렉토리 (--source-dir)", lambda: args.source_dir, lambda: True))
    if args.destination_dir is not None:
        arg_print_definitions.append(("찾은 같은 사진을 모아둘 곳 (--destination-dir)", lambda: args.destination_dir, lambda: True))
    if args.target_dir is not None:
        arg_print_definitions.append(("같은 사진이 있는지 찾을곳 (--target-dir)", lambda: args.target_dir, lambda: True))

    # 실제로 출력될 항목들만 필터링
    items_to_print = []
    for label, value_func, condition_func in arg_print_definitions:
        if condition_func():
            items_to_print.append((label, value_func))

    # 출력될 레이블들의 최대 시각적 길이 계산
    if not items_to_print:
        max_label_vl = 0
    else:
        max_label_vl = max(visual_length(label) for label, _ in items_to_print)

    # 값이 시작될 목표 시각적 컬럼 위치 설정
    # (앞공백 "  "의 시각적 길이 + 가장 긴 레이블의 시각적 길이 + 최소 하이픈 3개의 시각적 길이)
    target_value_start_column_vl = visual_length("  ") + max_label_vl + 3

    for label_text, value_func in items_to_print:
        value = value_func()
        prefix_with_spaces = f"  {label_text}"
        prefix_vl = visual_length(prefix_with_spaces)
        
        # 필요한 하이픈 개수 계산 (시각적 길이에 기반)
        num_hyphens_vl_needed = target_value_start_column_vl - prefix_vl
        num_hyphens = max(1, int(num_hyphens_vl_needed)) # 최소 1개의 하이픈 보장
        
        print(f"{prefix_with_spaces}{'-' * num_hyphens}{value}")
    print(f"--------------------------------------------------\n")
    return args

import unicodedata

# === 공유 로거 인스턴스 ===
# 이 logger 인스턴스를 다른 모듈에서 import하여 사용합니다.
logger = SimpleLogger()

# === 사용 예시 ===
if __name__ == "__main__":
    # 0. 현재 작업 디렉토리 및 스크립트 위치 정보 출력 (테스트용)
    direction_dir = os.getcwd()
    print(f"지금 쥔계서 계신곳(direction_dir)      : {direction_dir}")
    worker_path_obj = Path(__file__).resolve()
    project_root_path = worker_path_obj.parent.parent
    print(f"지금 일꾼이 일하는곳(worker_dir_name)  : {project_root_path}")

   # 1. 명령줄 인자 파싱
    # 이 함수는 --root-dir, --config-path, --log-dir, --log-level 인자를 처리합니다.
    args = get_argument()
    print(f"파싱된 인자: root_dir='{args.root_dir}', config_path='{args.config_path}', log_dir='{args.log_dir}', log_level='{args.log_level}'")

    # 2. 로거 설정
    # 로그 파일 이름에 타임스탬프를 추가하여 실행 시마다 고유한 로그 파일을 생성합니다.
    # 이 부분은 SimpleLogger 클래스 내부의 _apply_standalone_config 또는 외부 setup 호출로 대체될 수 있습니다.
    log_file_name = f"SimpleLogger_standalone_test_{datetime.now().strftime('%y%m%d_%H%M%S')}.log" # 모듈 이름 포함
    # args.log_dir은 get_argument에 의해 기본값(<root_dir>/logs) 또는 명령줄 값으로 설정됩니다.
    log_file_path = os.path.join(args.log_dir, log_file_name)

    logger = SimpleLogger(min_level="DEBUG", pretty_print=True, standalone=True)
    logger.info(f"--- json_manager.py standalone test execution ---")
    logger.info(f"로그 파일 경로: {log_file_path}")
    logger.show_config()

    # 명령줄 인자를 사용하여 로거 재설정
    logger.setup(
        logger_path=log_file_path,
        min_level=args.log_level,
        include_function_name=True, # 테스트 실행 시 함수 이름 포함이 유용합니다.
        pretty_print=True
    )

    # --- 로거 기능 테스트 함수 정의 ---
    print("\n--- Test 1: 기본 설정 (standalone에 의해 이미 DEBUG, 파일 로깅) 및 pretty print ---")
    def test_1():
        # 이 함수는 로거의 기본 동작과 pretty print 기능을 테스트합니다.
        def example_function_1():
            logger.info("첫 번째 함수에서 보낸 INFO 메시지")
            logger.debug("이 메시지는 기본 설정에서는 보이지 않습니다 (DEBUG 레벨)")

            sample_dict = {"status": "warning", "code": 400}
            # 수정된 부분: 딕셔너리와 문자열을 하나로 합쳐서 전달
            logger.warning(f"복잡한 객체 경고 메시지: {sample_dict}") # 또는 다른 방법 적용

        example_function_1()

    def test_2():
        print("\n--- 동기 파일 기록 및 상세 로깅 설정 ---")
        # 로거 설정을 변경하여 동기 파일 기록, DEBUG 레벨, 함수 이름 포함을 활성화합니다.
        log_file_path_sync = "./logs_sync/my_application_sync.log"
        logger.setup(logger_path=log_file_path_sync, min_level="DEBUG", include_function_name=True, async_file_writing=False)
        logger.show_config()

        def example_function_2():
            logger.debug("두 번째 함수에서 보낸 DEBUG 메시지 (이제 보임)")
            logger.info("두 번째 함수에서 보낸 INFO 메시지")
            data_list = [1, 2, 3, {"a": 1, "b": 2}]
            log_info = {
                "description": "오류 목록 메시지",
                "data": data_list
            }
            logger.error(log_info) # 이 새로운 딕셔너리 전체를 message로 전달        logger.error([1, 2, 3, {"a": 1, "b": 2}], "오류 목록 메시지") # 함수 이름 포함 및 Pretty print 적용됨

        example_function_2()

    print("\n--- Test 3: 비동기 파일 기록, INFO 레벨, 함수 이름 미포함 ---")
    def async_test_3():
        # 비동기 파일 기록 기능을 테스트합니다.
        time.sleep(0.1) # 이전 동기 쓰기가 완료될 시간을 약간 줍니다.

        print("\n--- 비동기 파일 기록 설정 ---")
        log_file_path_async = "./logs_async/my_application_async.log"
        # 여기서는 명시적으로 logger_path와 async_file_writing 모두 설정합니다.
        logger.setup(logger_path=log_file_path_async, min_level="INFO", include_function_name=False, async_file_writing=True)
        logger.show_config()

        def example_function_3():
            logger.info("세 번째 함수에서 보낸 INFO 메시지 (비동기 파일 기록)")
            sample_data = {"action": "process", "id": 100}
            logger.debug(sample_data) # DEBUG 레벨이라 파일에 기록 안됨 (min_level이 INFO)
            logger.critical("세 번째 함수에서 보낸 CRITICAL 메시지") # 파일에 기록됨

        example_function_3()

    def async_test_4():

        # 파일 기록을 비활성화하고 비동기 스레드가 정상적으로 종료되는지 테스트합니다.
        print("\n--- 파일 기록 비활성화 (비동기 스레드 종료) ---")
        logger.setup(logger_path=None) # logger_path를 None으로 설정하면 비동기 쓰레드도 종료됨
        logger.show_config()
        logger.info("파일 기록 비활성화 후 메시지 (콘솔에만 출력)")

        # 프로그램 종료 전에 logger.shutdown()을 호출하여
        # 비동기 스레드가 안전하게 모든 로그를 처리하고 종료되도록 합니다.
        print("\n프로그램 종료 전 로거 shutdown 호출...")
        logger.shutdown() # 비동기 스레드가 있다면 안전하게 종료

        # 참고: 실제 파일을 확인하시려면 스크립트 실행 후 ./logs_sync/ 또는 ./logs_async/ 디렉토리 안의 파일을 열어보세요.

    # 정의된 테스트 함수들 실행
    test_1()
    test_2()
    async_test_3()
    async_test_4()