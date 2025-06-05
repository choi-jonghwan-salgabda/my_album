import os
import shutil
from pathlib import Path

def remove_empty_directories(directory_path: Path, dry_run: bool = False, delete_top_if_empty: bool = False):
    """
    지정된 디렉토리 내의 모든 빈 하위 디렉토리를 재귀적으로 삭제합니다.

    Args:
        directory_path (Path): 검사 및 삭제를 시작할 디렉토리 경로.
        dry_run (bool, optional): True이면 실제 삭제 없이 삭제될 디렉토리만 출력합니다. 기본값은 False.
        delete_top_if_empty (bool, optional): True이고 최상위 directory_path도 비게 되면 삭제합니다.
                                             기본값은 False (최상위 디렉토리는 남김).
    Returns:
        bool: 최상위 directory_path가 비어있거나 삭제되었으면 True, 아니면 False.
    """
    if not directory_path.is_dir():
        print(f"오류: '{directory_path}'는 유효한 디렉토리가 아닙니다.")
        return False

    is_top_directory_empty_initially = not any(directory_path.iterdir())

    # 하위 디렉토리부터 재귀적으로 처리 (깊이 우선 탐색의 후위 순회 방식)
    for item in list(directory_path.iterdir()): # list()로 감싸서 순회 중 삭제 문제 방지
        if item.is_dir():
            # 재귀 호출의 결과 (하위 디렉토리가 비었는지 여부)는 여기서는 직접 사용하지 않음
            # remove_empty_directories 함수가 내부적으로 하위 빈 디렉토리를 삭제함
            remove_empty_directories(item, dry_run, delete_top_if_empty=True) # 하위는 항상 비면 삭제 시도

    # 현재 디렉토리가 비었는지 다시 확인 (하위 빈 디렉토리들이 삭제된 후)
    if not any(directory_path.iterdir()):
        try:
            if dry_run:
                print(f"(Dry Run) 빈 디렉토리 삭제 예정: '{directory_path}'")
            else:
                directory_path.rmdir() # 비어있으면 삭제
                print(f"빈 디렉토리 삭제됨: '{directory_path}'")
            return True # 디렉토리가 비었거나 (dry_run) 삭제됨
        except OSError as e:
            print(f"오류: 디렉토리 '{directory_path}' 삭제 중 오류 발생: {e}")
            return False # 삭제 실패
    return False # 디렉토리가 비어있지 않음

if __name__ == '__main__':
    # --- 테스트를 위한 설정 ---
    test_root_dir = Path("./test_empty_dir_removal")
    
    # 테스트용 디렉토리 구조 생성
    # test_root_dir/
    # ├── empty_top/
    # ├── level1_empty/
    # ├── level1_not_empty/
    # │   ├── file1.txt
    # │   └── level2_empty/
    # └── level1_with_deeper_empty/
    #     └── level2_also_empty/
    #         └── level3_empty/

    def setup_test_directories(root):
        if root.exists():
            shutil.rmtree(root) # 기존 테스트 디렉토리 삭제
        
        (root / "empty_top").mkdir(parents=True, exist_ok=True)
        (root / "level1_empty").mkdir(parents=True, exist_ok=True)
        
        level1_not_empty = root / "level1_not_empty"
        level1_not_empty.mkdir(parents=True, exist_ok=True)
        (level1_not_empty / "file1.txt").write_text("hello")
        (level1_not_empty / "level2_empty").mkdir(parents=True, exist_ok=True)
        
        level3_empty = root / "level1_with_deeper_empty" / "level2_also_empty" / "level3_empty"
        level3_empty.mkdir(parents=True, exist_ok=True)
        
        print(f"테스트 디렉토리 생성 완료: {root}")

    # 1. Dry run 테스트
    print("\n--- 1. Dry Run 테스트 (최상위 디렉토리 삭제 안 함) ---")
    setup_test_directories(test_root_dir)
    remove_empty_directories(test_root_dir, dry_run=True, delete_top_if_empty=False)
    if test_root_dir.exists():
        print(f"Dry run 후 '{test_root_dir}'는 여전히 존재합니다.")
    else:
        print(f"오류: Dry run임에도 '{test_root_dir}'가 삭제되었습니다.")


    # 2. 실제 삭제 테스트 (최상위 디렉토리 삭제 안 함)
    print("\n--- 2. 실제 삭제 테스트 (최상위 디렉토리 삭제 안 함) ---")
    setup_test_directories(test_root_dir)
    remove_empty_directories(test_root_dir, dry_run=False, delete_top_if_empty=False)
    if test_root_dir.exists():
        print(f"실제 삭제 후 '{test_root_dir}'는 여전히 존재합니다 (의도된 동작).")
        print("남아있는 디렉토리 구조:")
        for p in sorted(test_root_dir.rglob("*")): # 정렬된 순서로 출력
            indent = "  " * (len(p.parts) - len(test_root_dir.parts))
            print(f"{indent}{p.name}{'/' if p.is_dir() else ''}")
    else:
        print(f"오류: '{test_root_dir}'가 삭제되었습니다 (delete_top_if_empty=False).")


    # 3. 실제 삭제 테스트 (최상위 디렉토리가 비면 삭제)
    # 이 테스트를 위해서는 test_root_dir 자체가 비도록 만들어야 함
    # 여기서는 empty_top 디렉토리를 대상으로 테스트
    print("\n--- 3. 실제 삭제 테스트 (대상 디렉토리가 비면 삭제) ---")
    setup_test_directories(test_root_dir)
    target_for_top_level_delete = test_root_dir / "empty_top"
    print(f"'{target_for_top_level_delete}' 디렉토리에 대해 최상위 삭제 옵션 테스트:")
    remove_empty_directories(target_for_top_level_delete, dry_run=False, delete_top_if_empty=True)
    if not target_for_top_level_delete.exists():
        print(f"삭제 후 '{target_for_top_level_delete}'는 삭제되었습니다 (의도된 동작).")
    else:
        print(f"오류: '{target_for_top_level_delete}'가 삭제되지 않았습니다.")

    # 4. 특정 경로를 지정하여 테스트
    print("\n--- 4. 특정 경로 지정 테스트 ---")
    setup_test_directories(test_root_dir)
    specific_path_to_clean = test_root_dir / "level1_with_deeper_empty"
    print(f"'{specific_path_to_clean}' 내부의 빈 디렉토리 정리:")
    remove_empty_directories(specific_path_to_clean, dry_run=False, delete_top_if_empty=True) # 이 경우 specific_path_to_clean도 삭제됨
    if not specific_path_to_clean.exists():
         print(f"삭제 후 '{specific_path_to_clean}'는 삭제되었습니다 (의도된 동작).")
    else:
        print(f"오류: '{specific_path_to_clean}'가 삭제되지 않았습니다.")
        print("남아있는 디렉토리 구조:")
        for p in sorted(specific_path_to_clean.rglob("*")):
            indent = "  " * (len(p.parts) - len(specific_path_to_clean.parts))
            print(f"{indent}{p.name}{'/' if p.is_dir() else ''}")


    # 테스트 후 생성된 디렉토리 정리 (선택 사항)
    # print("\n테스트 완료. 테스트 디렉토리 정리 중...")
    # if test_root_dir.exists():
    #     shutil.rmtree(test_root_dir)
    # print("테스트 디렉토리 정리 완료.")

