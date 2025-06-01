import os
import time
import cv2
import numpy as np
import json
from shapely.geometry import *
import torch
import subprocess
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from dataset import TotalText, Ctw1500Text, Icdar15Text, Mlt2017Text, TD500Text, TD500HUSTText, \
    ArtText, ArtTextJson, Mlt2019Text, Ctw1500Text_New, TotalText_New, Ctw1500Text_mid, TotalText_mid, TD500HUSTText_mid, CustomDataset
from network.textnet import TextNet
from cfglib.config import config as cfg, update_config, print_config
from cfglib.option import BaseOptions
from util.augmentation import BaseTransform
from util.visualize import visualize_detection, visualize_gt
from util.misc import to_device, mkdirs, rescale_result, get_cosine_map

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
# 파일 상단의 import 섹션에 추가
import math
from scipy.spatial import ConvexHull

def expand_polygon(polygon, expand_ratio=1.2):
    """
    폴리곤의 각 점을 중심으로부터 expand_ratio만큼 확장
    
    Args:
        polygon: numpy array, 폴리곤 좌표 배열 (N, 2) 형태
        expand_ratio: 확장 비율 (1보다 크면 확장, 1보다 작으면 축소)
    
    Returns:
        확장된 폴리곤 좌표 배열
    """
    if len(polygon) < 3:
        return polygon
        
    # 무게 중심 계산
    center = np.mean(polygon, axis=0)
    
    # 각 점을 중심으로부터 확장
    expanded = []
    for pt in polygon:
        # 중심에서 해당 점 방향으로의 벡터
        vector = pt - center
        # 벡터 확장
        expanded_pt = center + vector * expand_ratio
        expanded.append(expanded_pt)
        
    return np.array(expanded, dtype=np.int32)

def process_and_visualize_contours(img_show, contours, meta, idx, expand_ratio=1.2):
    """
    컨투어 확장, 처리, 시각화를 수행하는 함수
    
    Args:
        img_show: 원본 이미지
        contours: 감지된 컨투어 목록
        meta: 메타데이터
        idx: 현재 이미지 인덱스
        expand_ratio: 확장 비율 (기본값: 1.2)
    
    Returns:
        expanded_contours: 확장된 컨투어 목록
    """
    # 결과 시각화 디렉토리 생성
    vis_output_dir = os.path.join(cfg.vis_dir, '{}_enhanced'.format(cfg.exp_name))
    os.makedirs(vis_output_dir, exist_ok=True)
    
    # 원본 이미지 복사
    original_img = img_show.copy()
    
    # 원본 컨투어 시각화
    orig_contour_img = img_show.copy()
    cv2.polylines(orig_contour_img, contours, True, (0, 255, 0), 2)
    
    # 컨투어 확장
    expanded_contours = []
    for cont in contours:
        expanded_cont = expand_polygon(cont, expand_ratio)
        expanded_contours.append(expanded_cont)
    
    # 확장된 컨투어 시각화
    expanded_contour_img = img_show.copy()
    cv2.polylines(expanded_contour_img, expanded_contours, True, (0, 0, 255), 2)
    
    # 반투명 채우기 이미지 생성
    filled_contour_img = img_show.copy()
    overlay = filled_contour_img.copy()
    for cont in expanded_contours:
        cv2.fillPoly(overlay, [cont], (0, 200, 255))
    # 반투명 효과 적용
    cv2.addWeighted(overlay, 0.3, filled_contour_img, 0.7, 0, filled_contour_img)
    cv2.polylines(filled_contour_img, expanded_contours, True, (0, 0, 255), 2)
    
    # 2x2 그리드로 이미지 배치
    h, w = img_show.shape[:2]
    combined_img = np.zeros((h*2, w*2, 3), dtype=np.uint8)
    combined_img[:h, :w, :] = original_img  # 좌상단: 원본
    combined_img[:h, w:, :] = orig_contour_img  # 우상단: 원본 컨투어
    combined_img[h:, :w, :] = expanded_contour_img  # 좌하단: 확장된 컨투어
    combined_img[h:, w:, :] = filled_contour_img  # 우하단: 채워진 컨투어
    
    # 각 영역에 레이블 추가
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    font_color = (255, 255, 255)
    
    cv2.putText(combined_img, "Original", (10, 30), 
               font, font_scale, font_color, font_thickness)
    cv2.putText(combined_img, f"Original Contours ({len(contours)})", (w+10, 30), 
               font, font_scale, font_color, font_thickness)
    cv2.putText(combined_img, f"Expanded Contours ({len(expanded_contours)})", (10, h+30), 
               font, font_scale, font_color, font_thickness)
    cv2.putText(combined_img, "Filled Regions", (w+10, h+30), 
               font, font_scale, font_color, font_thickness)
    
    # 결과 저장
    img_filename = meta['image_id'][idx]
    if '.' in img_filename:
        img_filename = os.path.splitext(img_filename)[0]
    
    output_path = os.path.join(vis_output_dir, f"{img_filename}_enhanced.jpg")
    cv2.imwrite(output_path, combined_img)
    print(f"개선된 이미지 저장: {output_path}")
    
    return expanded_contours

def filter_contours(contours, min_area=50, max_area=None):
    """
    면적 기준으로 컨투어를 필터링하는 함수
    
    Args:
        contours: 컨투어 목록
        min_area: 최소 면적 (이보다 작은 컨투어는 제거)
        max_area: 최대 면적 (이보다 큰 컨투어는 제거)
    
    Returns:
        필터링된 컨투어 목록
    """
    filtered = []
    for cont in contours:
        area = cv2.contourArea(cont)
        if area < min_area:
            continue
        if max_area and area > max_area:
            continue
        filtered.append(cont)
    
    return filtered if filtered else contours  # 모두 필터링되면 원본 반환

def osmkdir(out_dir):
    import shutil
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)


def write_to_file(contours, file_path):
    """
    :param contours: [[x1, y1], [x2, y2]... [xn, yn]]
    :param file_path: target file path
    """
    # according to total-text evaluation method, output file shoud be formatted to: y0,x0, ..... yn,xn
    with open(file_path, 'w') as f:
        for cont in contours:
            cont = np.stack([cont[:, 0], cont[:, 1]], 1)
            if cv2.contourArea(cont) <= 0:
                continue
            cont = cont.flatten().astype(str).tolist()
            cont = ','.join(cont)
            f.write(cont + '\n')


def inference(model, test_loader, output_dir):
    print(f"결과 파일 저장 경로: {output_dir}")
    # 경로가 존재하는지 확인하고 필요하면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"출력 디렉토리 생성: {output_dir}")

    total_time = 0.
    if cfg.exp_name != "MLT2017" and cfg.exp_name != "ArT" and cfg.exp_name != "ArT_mid":
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    else:
        if not os.path.exists(output_dir):
            mkdirs(output_dir)
        if cfg.exp_name == "MLT2017":
            out_dir = os.path.join(output_dir, "{}_{}_{}_{}_{}".
                                   format(str(cfg.checkepoch), cfg.test_size[0],
                                          cfg.test_size[1], cfg.dis_threshold, cfg.cls_threshold))
            if not os.path.exists(out_dir):
                mkdirs(out_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    art_results = dict()
    
    for i, (image, meta) in enumerate(test_loader):
        # 디버깅: 입력 이미지 형태 출력
        print(f"Input image shape: {image.shape}")
        
        # 이미지 형식이 NHWC(마지막에 채널)인 경우 NCHW로 변환
        if len(image.shape) == 4 and image.shape[1] > 3 and image.shape[3] == 3:
            print(f"이미지 형식을 NCHW로 변환합니다.")
            image = image.permute(0, 3, 1, 2)
            print(f"변환 후 이미지 형태: {image.shape}")
        
        # 채널 수 확인 및 수정
        if image.shape[1] != 3:
            print(f"경고: 이미지 채널 수가 3이 아닙니다: {image.shape[1]}")
            if image.shape[1] > 3:
                image = image[:, :3, :, :]
        
        input_dict = dict()
        idx = 0  # test mode can only run with batch_size == 1
        H, W = meta['Height'][idx].item(), meta['Width'][idx].item()
        
        # 이미지를 GPU로 이동
        image = image.to(cfg.device)
        input_dict['img'] = image

        # get detection result
        start = time.time()
        with torch.no_grad():
            output_dict = model(input_dict)
            
        torch.cuda.synchronize()
        end = time.time()
        if i > 0:
            total_time += end - start
            fps = (i + 1) / total_time
        else:
            fps = 0.0

        print('detect {} / {} images: {}. ({:.2f} fps)'.format(i + 1, len(test_loader), meta['image_id'][idx], fps))

        # visualization
        img_show = image[idx].permute(1, 2, 0).cpu().numpy()
        img_show = ((img_show * cfg.stds + cfg.means) * 255).astype(np.uint8)

        if cfg.viz:
            try:
                gt_contour = []
                label_tag = meta['label_tag'][idx].int().cpu().numpy()
                for annot, n_annot in zip(meta['annotation'][idx], meta['n_annotation'][idx]):
                    if n_annot.item() > 0:
                        gt_contour.append(annot[:n_annot].int().cpu().numpy())

                gt_vis = visualize_gt(img_show, gt_contour, label_tag)
                
                # PyTorch 텐서가 단일 요소가 아닌 경우 발생할 수 있는 오류 처리
                try:
                    show_boundary, heat_map = visualize_detection(img_show, output_dict, meta=meta)
                    show_map = np.concatenate([heat_map, gt_vis], axis=1)
                    show_map = cv2.resize(show_map, (320 * 3, 320))
                    im_vis = np.concatenate([show_map, show_boundary], axis=0)

                    path = os.path.join(cfg.vis_dir, '{}_test'.format(cfg.exp_name), meta['image_id'][idx].split(".")[0]+".jpg")
                    cv2.imwrite(path, im_vis)
                    print(f"시각화 이미지 저장: {path}")
                except Exception as e:
                    print(f"시각화 중 오류 발생: {e}")
            except Exception as e:
                print(f"시각화 준비 중 오류 발생: {e}")

        contours = output_dict["py_preds"][-1].int().cpu().numpy()
        img_show, contours = rescale_result(img_show, contours, H, W)

        try:
            # 폴리곤 확장 및 시각화 수행
            print(f"원본 컨투어 개수: {len(contours)}")
            
            # 컨투어 개선 및 시각화
            expanded_contours = process_and_visualize_contours(
                img_show, 
                contours, 
                meta, 
                idx, 
                expand_ratio=1.1  # 확장 비율 조정 가능
            )
            
            # 확장된 컨투어 필터링
            filtered_contours = filter_contours(
                expanded_contours,
                min_area=45,  # 최소 면적
                max_area=None  # 최대 면적 제한 없음
            )
            
            print(f"처리된 컨투어 개수: {len(filtered_contours)}")
            
            # 처리된 컨투어로 업데이트
            contours = filtered_contours
            
        except Exception as e:
            print(f"컨투어 처리 및 시각화 중 오류 발생: {e}")

        #empty GPU cache
        torch.cuda.empty_cache()

        # write to file
        if cfg.exp_name == "Icdar2015":
            fname = "res_" + meta['image_id'][idx].replace('jpg', 'txt')
            contours = data_transfer_ICDAR(contours)
            output_path = os.path.join(output_dir, fname)
            write_to_file(contours, output_path)
            print(f"결과 파일 저장: {output_path}")
            
        elif cfg.exp_name == "MLT2017":
            fname = meta['image_id'][idx].split("/")[-1].replace('ts', 'res')
            fname = fname.split(".")[0] + ".txt"
            output_path = os.path.join(out_dir, fname)
            data_transfer_MLT2017(contours, output_path)
            print(f"결과 파일 저장: {output_path}")
            
        elif cfg.exp_name == "TD500":
            fname = "res_" + meta['image_id'][idx].split(".")[0]+".txt"
            output_path = os.path.join(output_dir, fname)
            data_transfer_TD500(contours, output_path)
            print(f"결과 파일 저장: {output_path}")
            
        elif cfg.exp_name == "TD500HUST" or cfg.exp_name == "TD500HUST_mid":
            fname = "res_" + meta['image_id'][idx].split(".")[0]+".txt"
            output_path = os.path.join(output_dir, fname)
            data_transfer_TD500HUST(contours, output_path)
            print(f"결과 파일 저장: {output_path}")
            
        elif cfg.exp_name == "ArT" or cfg.exp_name == "ArT_mid":
            fname = meta['image_id'][idx].split(".")[0].replace('gt', 'res')
            art_result = []
            for j in range(len(contours)):
                art_res = dict()
                S = cv2.contourArea(contours[j], oriented=True)
                if S < 0:
                    art_res['points'] = contours[j].tolist()[::-1]
                else:
                    print((meta['image_id'], S))
                    continue
                art_res['confidence'] = float(output_dict['confidences'][j])
                art_result.append(art_res)
            art_results[fname] = art_result
            
        else:
            # 커스텀 데이터셋 등 다른 데이터셋 처리
            tmp = np.array(contours).astype(np.int32)
            # 차원 확인 후 처리
            if len(tmp.shape) < 3:
                # (n, 4, 2) 형태로 만들기
                if len(tmp) > 0 and len(tmp[0]) > 0:
                    tmp = np.array([np.reshape(c, (-1, 2)) for c in tmp])
                else:
                    tmp = np.array([])
    
            # 파일 이름에서 확장자 처리
            fname = meta['image_id'][idx]
            if '.' in fname:
                fname = os.path.splitext(fname)[0] + '.txt'
            else:
                fname = fname + '.txt'
                
            output_path = os.path.join(output_dir, fname)
            print(f"결과 파일 저장: {output_path}")
            write_to_file(contours, output_path)

    # ArT 결과 저장 (루프 밖에서 한 번만 실행)
    if cfg.exp_name == "ArT" or cfg.exp_name == "ArT_mid":
        json_path = os.path.join(output_dir, 'art_test_{}_{}_{}_{}_{}.json'.format(
            cfg.checkepoch, cfg.test_size[0], cfg.test_size[1],
            cfg.dis_threshold, cfg.cls_threshold))
        with open(json_path, 'w') as f:
            json.dump(art_results, f)
        print(f"ArT 결과 JSON 저장: {json_path}")
            
    # MLT2017 결과 저장
    elif cfg.exp_name == "MLT2017":
        father_path = "{}_{}_{}_{}_{}".format(str(cfg.checkepoch), cfg.test_size[0],
                                          cfg.test_size[1], cfg.dis_threshold, cfg.cls_threshold)
        subprocess.call(['sh', './output/MLT2017/eval_zip.sh', father_path])
        
    print(f"\n모든 결과 처리 완료! 결과 디렉토리: {output_dir}")

def main(vis_dir_path):
    if not os.path.exists(vis_dir_path):
        os.makedirs(vis_dir_path, exist_ok=True)
        
    if cfg.img_root and os.path.exists(cfg.img_root):
        print(f"커스텀 데이터셋 사용: {cfg.img_root}")
        testset = CustomDataset(
            data_root=cfg.img_root,
            is_training=False,
            transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
        )
    elif cfg.exp_name == "Totaltext" or cfg.exp_name == "Totaltext_mid":
        testset = TotalText(
            data_root='data/total-text-mat',
            ignore_list=None,
            is_training=False,
            transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
        )
    elif cfg.exp_name == "Ctw1500" or cfg.exp_name == 'Ctw1500_mid':
        testset = Ctw1500Text(
            data_root='data/ctw1500',
            is_training=False,
            transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
        )
    elif cfg.exp_name == "TD500HUST" or cfg.exp_name == "TD500HUST_mid":
        testset = TD500HUSTText(
            data_root='data/',
            is_training=False,
            transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
        )
    elif cfg.exp_name == "ArT" or cfg.exp_name == "ArT_mid":
        testset = ArtTextJson(
            data_root='data/ArT',
            is_training=False,
            transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
        )
    else:
        print("{} is not justify".format(cfg.exp_name))
        return

    cudnn.benchmark = False
    # Data
    test_loader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

    # Model
    model = TextNet(is_training=False, backbone=cfg.net)
    if args.resume:
        model_path = args.resume
        print(f"직접 지정된 모델 경로 사용: {model_path}")
    else:
        model_path = os.path.join(cfg.save_dir, cfg.exp_name,
                                'MixNet_{}_{}.pth'.format(model.backbone_name, cfg.checkepoch))
        print(f"기본 모델 경로 사용: {model_path}")

    model.load_model(model_path)
    model.to(cfg.device)  # copy to cuda
    model.eval()
    
    with torch.no_grad():
        print('Start testing MixNet.')
        output_dir = os.path.join(cfg.output_dir, cfg.exp_name)
        inference(model, test_loader, output_dir)

    print("{} eval finished.".format(cfg.exp_name))
    print(f"\n결과 파일 위치:")
    print(f"텍스트 결과: {os.path.join(cfg.output_dir, cfg.exp_name)}")
    print(f"시각화 결과: {vis_dir_path}")


if __name__ == "__main__":
    # parse arguments
    option = BaseOptions()
    args = option.initialize()
    update_config(cfg, args)

    vis_dir = os.path.join(cfg.vis_dir, '{}_test'.format(cfg.exp_name))
    # main
    main(vis_dir)