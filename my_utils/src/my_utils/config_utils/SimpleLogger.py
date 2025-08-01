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

import os
import pprint
import inspect
import queue # queue 모듈 임포트
import threading # threading 모듈 임포트
import tempfile # tempfile 모듈 임포트
import time # 필요에 따라 스레드 대기 등에 사용될 수 있습니다.
import sys
import yaml
import shutil
import textwrap
from pathlib import Path
from datetime import datetime
import traceback
from tqdm import tqdm # tqdm 진행률 바와의 호환성을 위해 임포트
from collections import deque
from typing import List, Tuple, Optional

try:
    from my_utils.config_utils.arg_utils import get_argument
except ImportError as e:
    print(f"치명적 오류: my_utils를 임포트할 수 없습니다. PYTHONPATH 및 의존성을 확인해주세요: {e}")
    sys.exit(1)


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

# setup 메서드에서 인자가 전달되었는지 여부를 확인하기 위한 Sentinel 객체입니다.
_SETUP_SENTINEL = object()

class SimpleLogger:
    """
    콘솔 출력 및 파일 기록(동기/비동기)을 지원하는 간단한 로깅 클래스입니다.
    로그 레벨, 호출 함수 이름 포함 여부, 딕셔너리/리스트의 보기 좋은 출력(pretty print) 등을 설정할 수 있습니다.
    """
    def __init__(self, file_min_level="INFO", pretty_print=True, standalone=False):
        self._file_min_level = "DEBUG"       # 기본값: 파일에는 전체 저장

        self._print_func = print  # tqdm-aware 로그가 아닐 경우 기본 출력

        # 로거 초기화
        self.LOG_LEVELS = LOG_LEVELS
        self._logger_path = None
        self._console_min_level = "WARNING"  # 기본값: 화면에는 warning 이상
        self._file_min_level = self._validate_level(file_min_level)
        self._file_min_level_int = self.LOG_LEVELS[self._file_min_level]
        self._include_function_name = False
        self._pretty_print = pretty_print

        self._async = False
        self._async_file_writing_enabled = False # 비동기 파일 쓰기 활성화 여부
        self._log_queue: Optional[queue.Queue] = None # 비동기 로깅 시 사용할 메시지 큐
        self._stop_event = threading.Event()     # 비동기 쓰기 스레드 종료 이벤트
        self._writing_thread = None
        self._writer_thread = None               # 비동기 쓰기 스레드 객체
        self._file_handle = None                 # 파일 핸들 (비동기 쓰기 스레드에서 사용)
        self._initialized = False                # 로거가 독립 실행 모드로 초기화되었는지 여부
        self._tqdm_aware = False                 # tqdm 호환 출력 모드 활성화 여부
        self._log_queue_max_size = 10000         # 비동기 로그 큐의 최대 크기
        self._log_queue_full_warning_sent = False # 로그 큐가 가득 찼다는 경고를 보냈는지 여부
        self._max_log_line_width = 0             # 로그 메시지 최대 너비 (0은 비활성화)

        # --- Log Rotation 기능 추가 ---
        self._log_rotation_max_bytes = 0  # 로그 회전 파일 최대 크기 (0은 비활성화)
        self._log_rotation_backup_count = 5 # 유지할 백업 로그 파일 수
        self._log_rotation_lock = threading.Lock() # 로그 회전 시 스레드 안전성을 위한 잠금

        # --- DiskSpaceMonitor 기능 통합 ---
        # DiskSpaceMonitor의 상태를 SimpleLogger의 속성으로 직접 관리합니다.
        self._disk_threshold_percent = 80.0  # 정밀 모니터링 시작 임계값 (%)
        self._disk_check_interval_secs = 10    # 임계값 미만일 때 재확인 간격 (초)
        self._disk_monitoring_devices = {}     # {dev_id: {'monitoring': bool, 'last_check': float}}
        self._disk_monitor_lock = threading.Lock() # 스레드 안전성을 위한 잠금
        # --- 통합 완료 ---

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
            return self._file_min_level
        upper_level = str(level_str).upper()
        if upper_level in self.LOG_LEVELS:
            # 입력 레벨이 None이면 현재 레벨 유지
            return upper_level
        print(f"경고: 유효하지 않은 로그 레벨 '{level_str}'입니다. 현재 설정 '{self._file_min_level}'을 유지합니다.")
        return self._file_min_level

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
        log_level = self._file_min_level  # ✅ 올바르게 참조
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
                self._file_min_level = log_cfg.get("level", log_level)
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
            console_min_level='WARNING',  # 콘솔엔 WARNING 이상만
            file_min_level=self._file_min_level,
            include_function_name=True,
            pretty_print=True,
            async_file_writing=False
        )
        print(f"독립 실행용 로깅 설정 완료: {log_path}")
        self._initialized = True # 독립 실행 설정 완료 플래그

    def setup(self, 
            logger_path: Optional[str] = _SETUP_SENTINEL, 
            include_function_name: Optional[bool] = None, 
            pretty_print: Optional[bool] = None, 
            async_file_writing: Optional[bool] = None, 
            console_min_level: Optional[str] = None, 
            file_min_level: Optional[str] = None,
            disk_monitor_threshold_percent: Optional[float] = None,
            disk_monitor_check_interval_secs: Optional[int] = None,
            log_queue_max_size: Optional[int] = None,
            log_rotation_max_bytes: Optional[int] = None,
            log_rotation_backup_count: Optional[int] = None,
            log_line_max_width: Optional[int] = None
        ):
        """
        로거 설정을 변경합니다. 각 인자는 None이 아닌 경우에만 해당 설정을 업데이트합니다.

        Args:
            logger_path (str): 로그 파일 경로 (예: './logs/app.log').
            file_min_level (str): 파일로 저장할 최소 로그 레벨. (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            console_min_level (str): 콘솔에 출력할 최소 로그 레벨.
            include_function_name (bool): 로그 메시지에 호출 함수 이름 포함 여부.
            pretty_print (bool): dict 또는 list 객체를 보기 좋게 출력할지 여부.
            disk_monitor_threshold_percent (float, optional): 디스크 사용률이 이 값 이상일 경우 정밀 감시를 시작합니다.
            disk_monitor_check_interval_secs (int, optional): 디스크 사용량을 다시 확인하는 간격 (초).
            async_file_writing (bool, optional): 비동기 파일 기록 사용 여부.
            log_queue_max_size (int, optional): 비동기 로그 큐의 최대 크기를 설정합니다.
            log_rotation_max_bytes (int, optional): 로그 파일을 회전시킬 최대 크기(바이트). 0이면 비활성화.
            log_rotation_backup_count (int, optional): 유지할 백업 로그 파일의 수.
            log_line_max_width (int, optional): 로그 메시지의 최대 너비. 초과 시 자동 줄바꿈. 0은 비활성화.

        Returns:
            None
        """
        old_logger_path = self._logger_path
        old_async_state = self._async_file_writing_enabled

        # 1. 새로운 설정을 결정합니다. 인자가 전달되지 않았으면 기존 값을 유지합니다.
        new_logger_path = logger_path if logger_path is not _SETUP_SENTINEL else old_logger_path
        desired_async_state = async_file_writing if async_file_writing is not None else self._async_file_writing_enabled

        # 2. 비동기 쓰기 스레드의 상태를 관리합니다.
        # 스레드를 중지해야 하는 경우: (1) 현재 켜져 있고 (2) 끄라는 요청이 있거나, 경로가 없어지거나, 경로가 바뀔 때
        if old_async_state and (not desired_async_state or new_logger_path is None or new_logger_path != old_logger_path):
            self._stop_async_writer()

        # 3. 로거의 속성들을 업데이트합니다.
        if logger_path is not _SETUP_SENTINEL:
            self._logger_path = logger_path

        if file_min_level is not None:
            self._file_min_level = self._validate_level(file_min_level)
            self._file_min_level_int = self.LOG_LEVELS[self._file_min_level]
        if include_function_name is not None:
            self._include_function_name = bool(include_function_name)
        if pretty_print is not None:
            self._pretty_print = bool(pretty_print)
        if console_min_level is not None:
            self._console_min_level = self._validate_level(console_min_level)

        # 4. 비동기 쓰기 스레드를 시작해야 하는 경우 시작합니다.
        # 스레드가 꺼져 있고, 켜라는 요청이 있으며, 경로가 유효할 때
        if desired_async_state and new_logger_path is not None and not self._async_file_writing_enabled:
            self._start_async_writer(new_logger_path)

        # 5. 비동기 쓰기가 요청되었으나 경로가 없는 경우 경고
        if desired_async_state and new_logger_path is None:
            print("경고: 비동기 파일 기록을 요청했지만 로그 파일 경로가 지정되지 않았습니다. 비동기 기록을 활성화할 수 없습니다.")

        # --- Disk Monitor 설정 업데이트 ---
        if disk_monitor_threshold_percent is not None:
            self._disk_threshold_percent = float(disk_monitor_threshold_percent)
        if disk_monitor_check_interval_secs is not None:
            self._disk_check_interval_secs = int(disk_monitor_check_interval_secs)
        
        if log_queue_max_size is not None:
            self._log_queue_max_size = int(log_queue_max_size)
        
        # --- Log Rotation 설정 업데이트 ---
        if log_rotation_max_bytes is not None:
            self._log_rotation_max_bytes = int(log_rotation_max_bytes)
        if log_rotation_backup_count is not None:
            self._log_rotation_backup_count = int(log_rotation_backup_count)
        if log_line_max_width is not None:
            self._max_log_line_width = int(log_line_max_width)

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
            # 파일 핸들을 스레드 내부에 유지합니다.
            self._file_handle = None
            # 로그 디렉토리가 없으면 생성
            log_dir = os.path.dirname(log_file_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            self._file_handle = open(log_file_path, 'a', encoding='utf-8', errors='surrogateescape')

            while not self._stop_event.is_set():
                try:
                    # 큐에서 메시지 가져오기 (짧은 타임아웃 사용)
                    message = self._log_queue.get(timeout=0.1)

                    if message is _SENTINEL:
                        # 종료 Sentinel을 받으면 루프 종료
                        print("비동기 로깅 스레드: 종료 Sentinel 수신.")
                        break
                    
                    # --- 로그 회전 확인 ---
                    if self._log_rotation_max_bytes > 0:
                        # 현재 파일 크기가 임계값을 넘으면 회전
                        # 메시지 길이도 고려하여 회전 여부 결정
                        if self._file_handle.tell() + len(message.encode('utf-8', 'surrogateescape')) >= self._log_rotation_max_bytes:
                            self._file_handle.close() # 현재 파일 핸들 닫기
                            self._rotate_log_file() # 파일 회전
                            # 회전 후 새 파일 핸들 열기
                            self._file_handle = open(log_file_path, 'a', encoding='utf-8', errors='surrogateescape')

                    # 메시지 파일에 쓰기
                    self._file_handle.write(message + '\n')
                    self._file_handle.flush() # 매 메시지마다 파일 버퍼를 비워 디스크에 즉시 기록
                    self._log_queue.task_done() # 큐 작업 완료 알림

                except queue.Empty:
                    # 큐가 비어있으면 타임아웃 발생. 루프 조건(stop_event) 다시 확인.
                    continue
                except Exception as e:
                    # 파일 쓰기 중 오류 발생 시 콘솔에 알림
                    print(f"오류: 비동기 쓰기 스레드에서 파일 '{log_file_path}'에 쓰기 실패: {e}")
                    if self._log_queue.qsize() > 0: # 오류 발생했더라도 메시지는 큐에서 제거
                            self._log_queue.task_done()

        except Exception as e:
            print(f"오류: 비동기 로깅 스레드 초기화 또는 파일 열기 실패: {e}")

        finally:
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
        self._log_queue = queue.Queue(maxsize=self._log_queue_max_size) # 최대 크기가 있는 큐 생성
        self._stop_event = threading.Event() # 스레드 종료를 위한 이벤트 객체 생성 (재사용 가능하도록)
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

    def set_tqdm_aware(self, use_tqdm_format: bool = True):
        """
        tqdm 사용 시 로그 메시지를 tqdm.write()로 출력하도록 설정합니다.
        use_tqdm_format=False일 경우 다시 print로 복원됩니다.
        """
        if use_tqdm_format:
            self._print_func = tqdm.write
        else:
            self._print_func = print

    def _print(self, msg):
        """
        tqdm 호환 로그 출력을 위해 내부적으로 사용하는 출력 함수입니다.
        tqdm-aware 모드에서는 tqdm.write를 사용하고, 그렇지 않으면 print를 사용합니다.
        """
        if hasattr(self, '_print_func'):
            self._print_func(msg)
        else:
            print(msg)  # 기본값

    # def set_tqdm_aware(self, aware: bool):
    #     """
    #     tqdm 진행률 바와 호환되는 출력 모드를 설정/해제합니다.
    #     True로 설정하면 print 대신 tqdm.write()를 사용합니다.
    #     """
    #     self._tqdm_aware = aware
    #     if aware:
    #         self.debug("tqdm 호환 출력 모드가 활성화되었습니다.")

    def get_config(self):
        """
        현재 로거의 주요 설정 값들을 딕셔너리 형태로 반환합니다.
        """
        return {
            "logger_path": self._logger_path, # 로그 파일 경로
            "file_min_level": self._file_min_level,
            "file_min_level_int": self._file_min_level_int,
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
        """
        로깅 함수를 호출한 실제 사용자 함수 이름을 가져옵니다.
        로거 내부의 래퍼 함수(예: debug, info, warning, error, critical, log, _format_message)를 건너뜁니다.
        """
        if not self._include_function_name:
            return None

        stack = inspect.stack()
        # 스택 구조: currentframe -> _get_caller_function_name -> log -> level_method -> user_function
        # user_function의 프레임은 currentframe 기준으로 4단계 위에 있습니다.
        # frame0: currentframe (_get_caller_function_name 내부)
        # frame1: _format_message 또는 log 호출
        # frame2: log 또는 level_method 호출
        # frame3: level_method 또는 user_function 호출
        # frame4: user_function (목표)

        caller_frame = None
        try:
            # 로거 파일의 절대 경로를 가져옵니다.
            logger_file_path = Path(__file__).resolve()

            # 스택을 역순으로 탐색하여 로거 파일 외부의 첫 번째 호출자를 찾습니다.
            # stack[0]은 _get_caller_function_name 자신입니다.
            for frame_info in stack:
                if Path(frame_info.filename).resolve() != logger_file_path:
                    return frame_info.function
            return "UnknownFunction" # 로거 외부의 호출자를 찾지 못한 경우
        except Exception as e:
            self.error(f"함수 이름 가져오기 오류: {e}", exc_info=True)
            return "ErrorGettingFunctionName"
        finally:
            # 스택 객체는 잠재적으로 큰 메모리를 차지하고 순환 참조를 유발할 수 있으므로,
            # 사용 후 명시적으로 삭제하여 메모리 누수를 방지하는 것이 좋습니다.
            del stack

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
        
        pid_str = f"{os.getpid():>5}"
        level_str = f"{level.upper():<8}" # 8자리로 왼쪽 정렬
        header_parts = [pid_str, timestamp, level_str]

        # 함수 이름 포함 설정이 켜져있으면 함수 이름을 가져와 헤더에 추가
        if self._include_function_name:
            # 함수 이름 포함 설정이 켜져있으면, 호출 함수 이름을 가져와 헤더에 추가
            function_name = self._get_caller_function_name()
            if function_name and function_name != '<module>': # 함수 이름을 성공적으로 가져왔고 메인 모듈이 아니면 추가
                header_parts.append(f"{function_name}")

        header = f"[{'|'.join(header_parts)}]"

        # 메시지 내용 포맷팅 (pretty_print 설정에 따름)
        if isinstance(message, (dict, list)) and self._pretty_print:
            # 메시지가 딕셔너리 또는 리스트이고 pretty_print가 활성화된 경우,
            # pprint.pformat을 사용하여 여러 줄로 보기 좋게 포맷합니다.
            message_content = pprint.pformat(message, indent=4, width=80, compact=False)
            formatted_output = f"{header}\n{message_content}"
        else:
            # 그 외 타입은 문자열로 변환
            message_content = str(message)

            # 자동 줄바꿈 기능 추가
            if self._max_log_line_width > 0:
                header_len = len(header) + 1  # 헤더 + 공백
                # 메시지가 들어갈 수 있는 최대 너비
                message_width = self._max_log_line_width - header_len

                if message_width > 10: # 줄바꿈을 적용하기에 너무 좁으면 의미 없음
                    lines = textwrap.wrap(message_content, width=message_width, break_long_words=False, replace_whitespace=False)
                    if len(lines) > 1:
                        first_line = f"{header} {lines[0]}"
                        # 나머지 줄들은 헤더 길이만큼 들여쓰기
                        other_lines = [" " * header_len + line for line in lines[1:]]
                        formatted_output = "\n".join([first_line] + other_lines)
                    else:
                        # 줄바꿈이 필요 없는 경우
                        formatted_output = f"{header} {message_content}"
                else:
                    # 줄바꿈을 적용하기에 너비가 너무 좁은 경우
                    formatted_output = f"{header} {message_content}"
            else:
                # 줄바꿈 기능이 비활성화된 경우
                formatted_output = f"{header} {message_content}"

        return formatted_output

    def _write_to_file_sync(self, formatted_message: str):
        """동기적으로 파일에 직접 씁니다."""
        if not self._logger_path:
            return
        try:
            # 디렉토리가 없으면 생성
            log_dir = os.path.dirname(self._logger_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            # 파일에 메시지 추가 모드로 쓰기
            with open(self._logger_path, 'a', encoding='utf-8', errors='surrogateescape') as f:
                f.write(formatted_message + '\n')
        except Exception as e:
            print(f"오류: 동기 모드에서 로그 파일 '{self._logger_path}'에 쓰기 실패: {e}")

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
                    # Non-blocking put을 사용하여 큐가 가득 찼을 때 프로그램이 멈추는 것을 방지
                    self._log_queue.put_nowait(formatted_message)
                    # 큐에 성공적으로 추가되면, 이전에 경고를 보냈더라도 이제 공간이 생겼을 수 있으므로 플래그를 리셋
                    self._log_queue_full_warning_sent = False
                except queue.Full:
                    # 큐가 가득 차서 현재 로그 메시지는 유실됩니다.
                    # 경고 메시지를 반복적으로 출력하지 않도록 플래그를 사용합니다.
                    if not self._log_queue_full_warning_sent:
                        warning_msg = f"경고: 비동기 로그 큐가 가득 찼습니다 (최대: {self._log_queue.maxsize}개). 일부 로그가 유실됩니다."
                        # 이 경고 메시지는 큐에 넣지 않고 직접 출력/기록합니다.
                        formatted_warning = self._format_message(warning_msg, "WARNING")
                        try:
                            self._print(formatted_warning)
                        except Exception:
                            pass # 오류 보고 중 오류 발생 방지
                        self._write_to_file_sync(formatted_warning)
                        self._log_queue_full_warning_sent = True
                except Exception as e:
                    print(f"오류: 비동기 큐에 메시지 추가 실패: {e}")
        else:
            # 동기 모드: 즉시 파일에 쓰기
            self._write_to_file_sync(formatted_message)

    def log(self, message, level="INFO"):
        level_str = self._validate_level(level)
        level_int = self.LOG_LEVELS[level_str]

        # 콘솔에 출력할지 판단
        if level_int >= self.LOG_LEVELS[self._console_min_level]:
            formatted_message = self._format_message(message, level_str)
            try:
                self._print(formatted_message)
            except UnicodeEncodeError:
                # 터미널 인코딩 문제 방지용
                encoding = sys.stdout.encoding or 'utf-8'
                safe_message = formatted_message.encode(encoding, errors='replace').decode(encoding)
                self._print(safe_message)

        # 파일에 기록할지 판단
        if level_int >= self.LOG_LEVELS[self._file_min_level]:
            formatted_message = self._format_message(message, level_str)
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

    def _rotate_log_file(self):
        """로그 파일을 회전시킵니다."""
        if not self._logger_path or not os.path.exists(self._logger_path):
            return

        with self._log_rotation_lock: # 회전 작업 중 다른 스레드가 파일에 접근하지 못하도록 잠금
            self.info(f"로그 파일 회전 시작: {self._logger_path}")
            # 백업 파일 이름 변경 (e.g., app.log.1 -> app.log.2, app.log.2 -> app.log.3)
            for i in range(self._log_rotation_backup_count - 1, 0, -1):
                sfn = f"{self._logger_path}.{i}"
                dfn = f"{self._logger_path}.{i + 1}"
                if os.path.exists(sfn):
                    if os.path.exists(dfn):
                        os.remove(dfn)
                    os.rename(sfn, dfn)

            # 현재 로그 파일을 첫 번째 백업 파일로 이름 변경 (e.g., app.log -> app.log.1)
            dfn = f"{self._logger_path}.1"
            if os.path.exists(dfn):
                os.remove(dfn)
            os.rename(self._logger_path, dfn)
            self.info(f"로그 파일이 회전되었습니다. 이전 로그는 {dfn} 에서 확인할 수 있습니다.")

    # --- DiskSpaceMonitor 통합 메서드 ---
    def _disk_get_device_info(self, path: Path) -> Tuple[int, Path]:
        """주어진 경로의 장치 ID와 해당 장치에 존재하는 경로를 가져옵니다."""
        try:
            p = path if path.exists() else path.parent
            while not p.exists():
                p = p.parent
            return os.stat(p).st_dev, p
        except Exception as e:
            self.error(f"'{path}' 경로의 장치 정보를 가져올 수 없습니다: {e}")
            root_path = Path('/')
            return os.stat(root_path).st_dev, root_path

    def _disk_update_monitoring_status(self, dev_id: int, existing_path_on_device: Path):
        """주기적으로 디스크 사용량을 확인하고 모니터링 상태를 동적으로 업데이트합니다."""
        now = time.time()
        dev_info = self._disk_monitoring_devices.get(dev_id, {'monitoring': False, 'last_check': 0})
        was_monitoring = dev_info.get('monitoring', False)

        # 마지막 확인 후 일정 시간이 지났으면 항상 재확인
        if now - dev_info.get('last_check', 0) > self._disk_check_interval_secs:
            try:
                usage = shutil.disk_usage(existing_path_on_device)
                percent_used = (usage.used / usage.total) * 100
                dev_info['last_check'] = now

                if percent_used >= self._disk_threshold_percent:
                    # 임계값을 넘으면 정밀 감시 활성화
                    dev_info['monitoring'] = True
                    if not was_monitoring:
                        # 이전에 감시 중이 아니었다면, 상태 변경을 알림
                        self.warning(f"장치 {dev_id}의 디스크 사용량이 {percent_used:.2f}%로, 임계값({self._disk_threshold_percent}%)을 초과했습니다. 이 장치에 대한 정밀 검사를 활성화합니다.")
                else:
                    # 임계값 미만이면 정밀 감시 비활성화
                    dev_info['monitoring'] = False
                    if was_monitoring:
                        # 이전에 감시 중이었다면, 상태 변경을 알림
                        self.info(f"장치 {dev_id}의 디스크 사용량이 {percent_used:.2f}%로, 임계값({self._disk_threshold_percent}%) 미만으로 회복되었습니다. 정밀 검사를 비활성화합니다.")
                
                self._disk_monitoring_devices[dev_id] = dev_info
            except Exception as e:
                self.error(f"장치 {dev_id}의 디스크 공간 확인 실패: {e}")
                # 오류 발생 시 안전을 위해 모니터링 상태를 유지하거나, 기본값으로 되돌릴 수 있음
                # 여기서는 마지막 상태를 유지하도록 별도 처리를 하지 않음
                dev_info['last_check'] = now # 오류가 났더라도 체크 시간은 업데이트하여 반복적인 오류 방지
                self._disk_monitoring_devices[dev_id] = dev_info

    def disk_pre_write_check(self, dest_path: Path, size_to_write: int):
        """파일을 쓰기 전에 디스크 공간이 충분한지 확인합니다."""
        with self._disk_monitor_lock: # 스레드 안전성 보장
            dev_id, existing_path_on_device = self._disk_get_device_info(dest_path)
            self._disk_update_monitoring_status(dev_id, existing_path_on_device)
            dev_info = self._disk_monitoring_devices.get(dev_id)
            if dev_info and dev_info['monitoring']:
                try:
                    usage = shutil.disk_usage(existing_path_on_device)
                    if usage.free < size_to_write:
                        raise DiskFullError(f"사전 검사 실패: '{dest_path}'에 {size_to_write} 바이트를 쓸 공간이 부족합니다. (남은 공간: {usage.free} 바이트)")
                except DiskFullError:
                    # 이 예외는 의도적으로 발생시킨 것이므로, 다시 발생시켜 호출자에게 전달합니다.
                    raise
                except Exception as e:
                    # shutil.disk_usage() 등에서 발생할 수 있는 다른 예외들은 로깅합니다.
                    self.error(f"디스크 공간 사전 검사 중 예상치 못한 오류 발생: {e}")
    # --- 통합 완료 ---

# === 공유 로거 인스턴스 ===
# 이 logger 인스턴스를 다른 모듈에서 import하여 사용합니다.
logger = SimpleLogger()

import unicodedata

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

    logger = SimpleLogger(file_min_level="DEBUG", pretty_print=True, standalone=True)
    logger.info(f"--- json_manager.py standalone test execution ---")
    logger.info(f"로그 파일 경로: {log_file_path}")
    logger.show_config()

    # 명령줄 인자를 사용하여 로거 재설정
    logger.setup(
        logger_path=log_file_path,
        console_min_level='WARNING',  # 콘솔엔 WARNING 이상만
        file_min_level=args.log_level,
        include_function_name=True, # 테스트 실행 시 함수 이름 포함이 유용합니다.
        pretty_print=True
    )

    # --- 로거 기능 테스트 함수 정의 ---
    def test_1():
        print("\n--- Test 1: 기본 설정 (standalone에 의해 이미 DEBUG, 파일 로깅) 및 pretty print ---")
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
        logger.setup(logger_path=log_file_path_sync, file_min_level="DEBUG", include_function_name=True, async_file_writing=False)
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

    def async_test_3():
        print("\n--- Test 3: 비동기 파일 기록, INFO 레벨, 함수 이름 미포함 ---")
        # 비동기 파일 기록 기능을 테스트합니다.
        time.sleep(0.1) # 이전 동기 쓰기가 완료될 시간을 약간 줍니다.

        print("\n--- 비동기 파일 기록 설정 ---")
        log_file_path_async = "./logs_async/my_application_async.log"
        # 여기서는 명시적으로 logger_path와 async_file_writing 모두 설정합니다.
        logger.setup(logger_path=log_file_path_async, file_min_level="INFO", include_function_name=False, async_file_writing=True)
        logger.show_config()

        def example_function_3():
            logger.info("세 번째 함수에서 보낸 INFO 메시지 (비동기 파일 기록)")
            sample_data = {"action": "process", "id": 100}
            logger.debug(sample_data) # DEBUG 레벨이라 파일에 기록 안됨 (file_min_level INFO)
            logger.critical("세 번째 함수에서 보낸 CRITICAL 메시지") # 파일에 기록됨

        example_function_3()

    def test_disk_space_monitor():
        print("\n--- Test: Disk Space Monitor and Safe File Operations ---")

        temp_dir = None
        original_threshold = logger._disk_threshold_percent # Store original threshold
        original_interval = logger._disk_check_interval_secs # Store original interval

        try:
            # 1. Create a temporary directory for the test
            temp_dir = Path(tempfile.mkdtemp())
            logger.info(f"임시 테스트 디렉토리 생성: {temp_dir}")

            # 2. Create a dummy source file
            dummy_src_file = temp_dir / "dummy_source.txt"
            with open(dummy_src_file, "wb") as f:
                f.write(os.urandom(1024 * 1024)) # 1MB dummy file
            logger.info(f"더미 소스 파일 생성: {dummy_src_file} ({dummy_src_file.stat().st_size} bytes)")

            # 3. Destination paths
            dummy_dst_file_copy = temp_dir / "dummy_destination_copy.txt"
            dummy_dst_file_move = temp_dir / "dummy_destination_move.txt"

            # --- Test 1: Normal operation (should succeed) ---
            logger.info("--- Test 1.1: safe_copy (정상 작동 확인) ---")
            try:
                safe_copy(str(dummy_src_file), str(dummy_dst_file_copy))
                logger.info(f"safe_copy 성공: {dummy_src_file} -> {dummy_dst_file_copy}")
                if not dummy_dst_file_copy.exists():
                    logger.error("safe_copy 정상 작동 테스트 실패: 대상 파일이 생성되지 않았습니다.")
            except DiskFullError as e:
                logger.error(f"safe_copy에서 예상치 못한 DiskFullError 발생 (정상 작동 테스트): {e}")
            except Exception as e:
                logger.error(f"safe_copy에서 예상치 못한 오류 발생 (정상 작동 테스트): {e}", exc_info=True)

            # Clean up for next test
            if dummy_dst_file_copy.exists():
                dummy_dst_file_copy.unlink()

            logger.info("--- Test 1.2: safe_move (정상 작동 확인) ---")
            # safe_copy는 원본 파일을 이동시키지 않으므로, dummy_src_file은 여전히 존재합니다.
            # 하지만 테스트의 견고성을 위해 다시 생성하여 깨끗한 상태를 보장합니다.
            with open(dummy_src_file, "wb") as f:
                f.write(os.urandom(1024 * 1024)) # 1MB dummy file
            try:
                safe_move(str(dummy_src_file), str(dummy_dst_file_move))
                logger.info(f"safe_move 성공: {dummy_src_file} -> {dummy_dst_file_move}")
                if not dummy_dst_file_move.exists():
                    logger.error("safe_move 정상 작동 테스트 실패: 대상 파일이 생성되지 않았습니다.")
                if dummy_src_file.exists(): # 원본 파일은 이동 후 삭제되어야 합니다.
                    logger.error("safe_move 정상 작동 테스트 실패: 원본 파일이 삭제되지 않았습니다.")
            except DiskFullError as e:
                logger.error(f"safe_move에서 예상치 못한 DiskFullError 발생 (정상 작동 테스트): {e}")
            except Exception as e:
                logger.error(f"safe_move에서 예상치 못한 오류 발생 (정상 작동 테스트): {e}", exc_info=True)

            # --- Test 2: DiskFullError 시뮬레이션 시도 (실제 디스크 공간에 따라 결과 다름) ---
            logger.info("--- Test 2: DiskFullError 시뮬레이션 시도 (실제 디스크 공간에 따라 결과 다름) ---")
            # 임계값을 매우 낮게 설정하여 모니터링 모드를 강제 활성화합니다.
            # 이렇게 하면 디스크 사용량이 조금만 있어도 '정밀 모니터링' 상태가 됩니다.
            logger._disk_threshold_percent = 0.0001
            logger._disk_check_interval_secs = 0.01 # 짧은 간격으로 확인하여 즉시 모니터링 상태 반영
            logger.setup(
                disk_monitor_threshold_percent=0.0001,
                disk_monitor_check_interval_secs=0.01
            )
            logger._disk_threshold_percent = 0.0001
            logger._disk_check_interval_secs = 0.01 # 짧은 간격으로 확인하여 즉시 모니터링 상태 반영
            logger._disk_monitoring_devices = {} # 이전 모니터링 상태 초기화

            # 재차 safe_copy 시도 (DiskFullError가 발생할 수 있는지 확인)
            # 이 테스트는 실제 디스크 공간이 부족하지 않으면 DiskFullError를 발생시키지 않습니다.
            # DiskFullError가 발생하면 성공, 발생하지 않으면 디스크 공간이 충분하다는 의미입니다.
            dummy_src_file_for_copy_test = temp_dir / "dummy_source_for_copy_test.txt"
            with open(dummy_src_file_for_copy_test, "wb") as f:
                f.write(os.urandom(1024 * 1024)) # 1MB dummy file

            try:
                logger.info(f"safe_copy 시도: {dummy_src_file_for_copy_test} -> {dummy_dst_file_copy} (DiskFullError 예상)")
                safe_copy(str(dummy_src_file_for_copy_test), str(dummy_dst_file_copy))
                logger.info(f"safe_copy (DiskFullError 예상) - 실제 디스크 공간이 충분하여 DiskFullError가 발생하지 않았습니다. (정상)")
            except DiskFullError as e:
                logger.info(f"safe_copy (DiskFullError 예상) - DiskFullError가 성공적으로 발생했습니다: {e}")
            except Exception as e:
                logger.error(f"safe_copy (DiskFullError 예상) - 예상치 못한 오류 발생: {e}", exc_info=True)

            # 재차 safe_move 시도 (DiskFullError가 발생할 수 있는지 확인)
            dummy_src_file_for_move_test = temp_dir / "dummy_source_for_move_test.txt"
            with open(dummy_src_file_for_move_test, "wb") as f:
                f.write(os.urandom(1024 * 1024)) # 1MB dummy file

            try:
                logger.info(f"safe_move 시도: {dummy_src_file_for_move_test} -> {dummy_dst_file_move} (DiskFullError 예상)")
                safe_move(str(dummy_src_file_for_move_test), str(dummy_dst_file_move))
                logger.info(f"safe_move (DiskFullError 예상) - 실제 디스크 공간이 충분하여 DiskFullError가 발생하지 않았습니다. (정상)")
            except DiskFullError as e:
                logger.info(f"safe_move (DiskFullError 예상) - DiskFullError가 성공적으로 발생했습니다: {e}")
            except Exception as e:
                logger.error(f"safe_move (DiskFullError 예상) - 예상치 못한 오류 발생: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"디스크 공간 모니터 테스트 중 최상위 오류 발생: {e}", exc_info=True)
        finally:
            # 원래 임계값과 간격으로 복원하고 모니터링 상태를 초기화합니다.
            logger.setup(
                disk_monitor_threshold_percent=original_threshold,
                disk_monitor_check_interval_secs=original_interval
            )
            logger._disk_monitoring_devices = {} # 모니터링 상태 초기화

            if temp_dir and temp_dir.exists():
                logger.info(f"임시 테스트 디렉토리 삭제: {temp_dir}")
                shutil.rmtree(temp_dir)
            logger.info("--- Disk Space Monitor 테스트 완료 ---")

    def test_log_rotation():
        print("\n--- Test: Log Rotation ---")
        log_dir = Path("./logs_rotation_test")
        if log_dir.exists():
            shutil.rmtree(log_dir)
        log_dir.mkdir()
        log_path = log_dir / "rotation_test.log"

        # 작은 크기로 로그 회전 설정
        logger.setup(
            logger_path=str(log_path),
            log_rotation_max_bytes=1024, # 1KB
            log_rotation_backup_count=3,
            file_min_level="DEBUG",
            async_file_writing=False # 동기 모드로 테스트하여 즉시 파일 크기 확인
        )
        logger.info("--- 로그 회전 테스트 시작 ---")

        # 1KB를 초과하는 로그 메시지 작성
        long_message = "A" * 200 # 200바이트 메시지
        for i in range(10): # 10 * 200 = 2000 바이트 (1KB 초과)
            logger.debug(f"로그 메시지 {i}: {long_message}")

        # 파일 존재 여부 확인
        if (log_dir / "rotation_test.log").exists() and \
           (log_dir / "rotation_test.log.1").exists():
            logger.info("[성공] 로그 회전이 성공적으로 수행되었습니다. rotation_test.log 와 rotation_test.log.1 파일이 생성되었습니다.")
        else:
            logger.error(f"[실패] 로그 회전이 수행되지 않았습니다. 파일 존재 여부: {(log_dir / 'rotation_test.log').exists()}, {(log_dir / 'rotation_test.log.1').exists()}")

        # 정리
        shutil.rmtree(log_dir)
        logger.info("로그 회전 테스트 디렉토리 정리 완료.")
        # 로거 설정을 기본값으로 되돌림
        logger.setup(logger_path=None, log_rotation_max_bytes=0) # 로그 회전 기능 비활성화

    def async_test_4():

        print("\n--- 파일 기록 비활성화 (비동기 스레드 종료) ---")
        # 파일 기록을 비활성화하고 비동기 스레드가 정상적으로 종료되는지 테스트합니다.
        logger.setup(logger_path=None) # logger_path를 None으로 설정하면 비동기 쓰레드도 종료됨
        logger.show_config()
        logger.info("파일 기록 비활성화 후 메시지 (콘솔에만 출력)")

        print("\n프로그램 종료 전 로거 shutdown 호출...")
        # 프로그램 종료 전에 logger.shutdown()을 호출하여
        # 비동기 스레드가 안전하게 모든 로그를 처리하고 종료되도록 합니다.
        logger.shutdown() # 비동기 스레드가 있다면 안전하게 종료

        # 참고: 실제 파일을 확인하시려면 스크립트 실행 후 ./logs_sync/ 또는 ./logs_async/ 디렉토리 안의 파일을 열어보세요.

    # 정의된 테스트 함수들 실행
    test_disk_space_monitor() # 새로운 테스트 함수 호출
    test_1()
    test_log_rotation() # 로그 회전 테스트 호출
    test_2()
    async_test_3()
    async_test_4()