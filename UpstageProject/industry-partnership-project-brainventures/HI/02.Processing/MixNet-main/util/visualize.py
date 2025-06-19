import torch
import numpy as np
import cv2
import os
import math
from cfglib.config import config as cfg
from util import canvas as cav
import matplotlib
matplotlib.use('agg')
import pylab as plt
from matplotlib import cm
import torch.nn.functional as F


def visualize_network_output(output_dict, input_dict, mode='train'):
    vis_dir = os.path.join(cfg.vis_dir, cfg.exp_name + '_' + mode)
    if not os.path.exists(vis_dir):
        os.mkdir(vis_dir)

    try:
        fy_preds = F.interpolate(output_dict["fy_preds"], scale_factor=cfg.scale, mode='bilinear')
        fy_preds = fy_preds.data.cpu().numpy()

        py_preds = output_dict["py_preds"][1:]
        init_polys = output_dict["py_preds"][0]
        inds = output_dict["inds"]

        if cfg.mid == True:
            midline = output_dict["midline"]

        image = input_dict['img']
        tr_mask = input_dict['tr_mask'].data.cpu().numpy() > 0
        distance_field = input_dict['distance_field'].data.cpu().numpy()
        direction_field = input_dict['direction_field']
        weight_matrix = input_dict['weight_matrix']
        gt_tags = input_dict['gt_points'].cpu().numpy()
        ignore_tags = input_dict['ignore_tags'].cpu().numpy()

        b, c, _, _ = fy_preds.shape
        for i in range(b):
            try:
                fig = plt.figure(figsize=(12, 9))

                mask_pred = fy_preds[i, 0, :, :]
                distance_pred = fy_preds[i, 1, :, :]
                norm_pred = np.sqrt(fy_preds[i, 2, :, :] ** 2 + fy_preds[i, 3, :, :] ** 2)
                angle_pred = 180 / math.pi * np.arctan2(fy_preds[i, 2, :, :], fy_preds[i, 3, :, :] + 0.00001)

                ax1 = fig.add_subplot(341)
                ax1.set_title('mask_pred')
                # ax1.set_autoscale_on(True)
                im1 = ax1.imshow(mask_pred, cmap=cm.jet)
                # plt.colorbar(im1, shrink=0.5)

                ax2 = fig.add_subplot(342)
                ax2.set_title('distance_pred')
                # ax2.set_autoscale_on(True)
                im2 = ax2.imshow(distance_pred, cmap=cm.jet)
                # plt.colorbar(im2, shrink=0.5)

                ax3 = fig.add_subplot(343)
                ax3.set_title('norm_pred')
                # ax3.set_autoscale_on(True)
                im3 = ax3.imshow(norm_pred, cmap=cm.jet)
                # plt.colorbar(im3, shrink=0.5)

                ax4 = fig.add_subplot(344)
                ax4.set_title('angle_pred')
                # ax4.set_autoscale_on(True)
                im4 = ax4.imshow(angle_pred, cmap=cm.jet)
                # plt.colorbar(im4, shrink=0.5)

                mask_gt = tr_mask[i]
                distance_gt = distance_field[i]
                # gt_flux = 0.999999 * direction_field[i] / (direction_field[i].norm(p=2, dim=0) + 1e-9)
                gt_flux = direction_field[i].cpu().numpy()
                norm_gt = np.sqrt(gt_flux[0, :, :] ** 2 + gt_flux[1, :, :] ** 2)
                angle_gt = 180 / math.pi * np.arctan2(gt_flux[0, :, :], gt_flux[1, :, :]+0.00001)

                ax11 = fig.add_subplot(345)
                # ax11.set_title('mask_gt')
                # ax11.set_autoscale_on(True)
                im11 = ax11.imshow(mask_gt, cmap=cm.jet)
                # plt.colorbar(im11, shrink=0.5)

                ax22 = fig.add_subplot(346)
                # ax22.set_title('distance_gt')
                # ax22.set_autoscale_on(True)
                im22 = ax22.imshow(distance_gt, cmap=cm.jet)
                # plt.colorbar(im22, shrink=0.5)

                ax33 = fig.add_subplot(347)
                # ax33.set_title('norm_gt')
                # ax33.set_autoscale_on(True)
                im33 = ax33.imshow(norm_gt, cmap=cm.jet)
                # plt.colorbar(im33, shrink=0.5)

                ax44 = fig.add_subplot(348)
                # ax44.set_title('angle_gt')
                # ax44.set_autoscale_on(True)
                im44 = ax44.imshow(angle_gt, cmap=cm.jet)
                # plt.colorbar(im44, shrink=0.5)

                img_show = image[i].permute(1, 2, 0).cpu().numpy()
                img_show = ((img_show * cfg.stds + cfg.means) * 255).astype(np.uint8)
                img_show = np.ascontiguousarray(img_show[:, :, ::-1])
                shows = []
                gt = gt_tags[i]
                gt_idx = np.where(ignore_tags[i] > 0)
                gt_py = gt[gt_idx[0], :, :]
                index = torch.where(inds[0] == i)[0]
                init_py = init_polys[index].detach().cpu().numpy()

                image_show = img_show.copy()
                cv2.drawContours(image_show, init_py.astype(np.int32), -1, (255, 0, 0), 2)
                cv2.drawContours(image_show, gt_py.astype(np.int32), -1, (0, 255, 255), 2)
                shows.append(image_show)
                for py in py_preds:
                    contours = py[index].detach().cpu().numpy()
                    image_show = img_show.copy()
                    cv2.drawContours(image_show, init_py.astype(np.int32), -1, (0, 125, 125), 2)
                    cv2.drawContours(image_show, gt_py.astype(np.int32), -1, (255, 125, 0), 2)
                    cv2.drawContours(image_show, contours.astype(np.int32), -1, (0, 255, 125), 2)
                    if cfg.mid == True:
                        cv2.polylines(image_show, midline.astype(np.int32), False, (125, 255, 0), 2)
                    shows.append(image_show)

                for idx, im_show in enumerate(shows):
                    axb = fig.add_subplot(3, 4, 9+idx)
                    # axb.set_title('boundary_{}'.format(idx))
                    # axb.set_autoscale_on(True)
                    im11 = axb.imshow(im_show, cmap=cm.jet)
                    # plt.colorbar(im11, shrink=0.5)

                path = os.path.join(vis_dir, '{}.png'.format(i))
                plt.savefig(path)
                plt.close(fig)
            except Exception as e:
                print(f"이미지 {i} 시각화 중 오류 발생: {str(e)}")
                continue
    except Exception as e:
        print(f"네트워크 출력 시각화 중 오류 발생: {str(e)}")


def visualize_gt(image, contours, label_tag):
    try:
        image_show = image.copy()
        image_show = np.ascontiguousarray(image_show[:, :, ::-1])

        # 안전하게 컨투어 그리기
        try:
            image_show = cv2.polylines(image_show,
                                    [contours[i] for i, tag in enumerate(label_tag) if tag > 0], True, (0, 0, 255), 3)
        except Exception as e:
            print(f"양성 컨투어 그리기 중 오류 발생: {str(e)}")
        
        try:
            image_show = cv2.polylines(image_show,
                                    [contours[i] for i, tag in enumerate(label_tag) if tag < 0], True, (0, 255, 0), 3)
        except Exception as e:
            print(f"음성 컨투어 그리기 중 오류 발생: {str(e)}")

        show_gt = cv2.resize(image_show, (320, 320))
        return show_gt
    
    except Exception as e:
        print(f"GT 시각화 중 오류 발생: {str(e)}")
        # 오류 발생 시 원본 이미지 반환
        try:
            return cv2.resize(image, (320, 320))
        except:
            # 최후의 수단으로 빈 이미지 반환
            return np.zeros((320, 320, 3), dtype=np.uint8)


def visualize_detection(image, output_dict, meta=None):
    try:
        image_show = image.copy()
        image_show = np.ascontiguousarray(image_show[:, :, ::-1])

        cls_preds = F.interpolate(output_dict["fy_preds"], scale_factor=cfg.scale, mode='bilinear')
        cls_preds = cls_preds[0].data.cpu().numpy()

        py_preds = output_dict["py_preds"][1:]
        init_polys = output_dict["py_preds"][0]
        shows = []

        if cfg.mid:
            midline = output_dict["midline"]

        init_py = init_polys.data.cpu().numpy()
        
        # 이미지 ID 안전하게 가져오기
        try:
            img_id = meta['image_id'][0].split(".")[0] if isinstance(meta['image_id'][0], str) else "unknown"
        except:
            img_id = "unknown"
            
        path = os.path.join(cfg.vis_dir, '{}_test'.format(cfg.exp_name), f"{img_id}_init.png")

        im_show0 = image_show.copy()
        for i, bpts in enumerate(init_py.astype(np.int32)):
            cv2.drawContours(im_show0, [bpts.astype(np.int32)], -1, (255, 0, 255), 2)
            for j, pp in enumerate(bpts):
                # 텐서인 경우 안전하게 변환
                if isinstance(pp[0], torch.Tensor):
                    x = int(pp[0].item())
                    y = int(pp[1].item())
                else:
                    x = int(pp[0])
                    y = int(pp[1])
                
                if j == 0:
                    cv2.circle(im_show0, (x, y), 3, (125, 125, 255), -1)
                elif j == 1:
                    cv2.circle(im_show0, (x, y), 3, (125, 255, 125), -1)
                else:
                    cv2.circle(im_show0, (x, y), 3, (255, 125, 125), -1)

        cv2.imwrite(path, im_show0)

        for idx, py in enumerate(py_preds):
            try:
                im_show = im_show0.copy()
                contours = py.data.cpu().numpy()
                cv2.drawContours(im_show, contours.astype(np.int32), -1, (0, 255, 255), 2)
                for ppts in contours:
                    for j, pp in enumerate(ppts):
                        # 텐서인 경우 안전하게 변환
                        if isinstance(pp[0], torch.Tensor):
                            x = int(pp[0].item())
                            y = int(pp[1].item())
                        else:
                            x = int(pp[0])
                            y = int(pp[1])
                            
                        if j == 0:
                            cv2.circle(im_show, (x, y), 3, (125, 125, 255), -1)
                        elif j == 1:
                            cv2.circle(im_show, (x, y), 3, (125, 255, 125), -1)
                        else:
                            cv2.circle(im_show, (x, y), 3, (255, 125, 125), -1)
                            
                if cfg.mid:
                    for ppt in midline:
                        for pt in ppt:
                            if isinstance(pt[0], torch.Tensor):
                                # 텐서인 경우 item() 메서드로 변환
                                x = int(pt[0].item())
                                y = int(pt[1].item())
                            else:
                                x = int(pt[0])
                                y = int(pt[1])

                            cv2.circle(im_show, (x, y), 3, (255, 0, 0), -1)

                iter_path = os.path.join(cfg.vis_dir, '{}_test'.format(cfg.exp_name),
                                f"{img_id}_{idx}iter.png")
                cv2.imwrite(iter_path, im_show)
                shows.append(im_show)
            except Exception as e:
                print(f"반복 {idx} 시각화 중 오류 발생: {str(e)}")
                # 오류 발생 시 원본 이미지 사용
                shows.append(im_show0.copy())

        # 안전하게 이미지 연결
        try:
            show_img = np.concatenate(shows, axis=1)
            show_boundary = cv2.resize(show_img, (320 * len(py_preds), 320))
        except Exception as e:
            print(f"이미지 연결 중 오류 발생: {str(e)}")
            # 오류 발생 시 첫 번째 이미지만 사용
            if shows:
                show_boundary = cv2.resize(shows[0], (320, 320))
            else:
                show_boundary = np.zeros((320, 320, 3), dtype=np.uint8)

        try:
            cls_pred = cav.heatmap(np.array(cls_preds[0] * 255, dtype=np.uint8))
            dis_pred = cav.heatmap(np.array(cls_preds[1] * 255, dtype=np.uint8))

            heat_map = np.concatenate([cls_pred*255, dis_pred*255], axis=1)
            heat_map = cv2.resize(heat_map, (320 * 2, 320))
        except Exception as e:
            print(f"히트맵 생성 중 오류 발생: {str(e)}")
            # 오류 발생 시 빈 히트맵 생성
            heat_map = np.zeros((320, 320 * 2, 3), dtype=np.uint8)

        return show_boundary, heat_map
        
    except Exception as e:
        print(f"시각화 중 오류 발생: {str(e)}")
        # 오류 발생 시 빈 이미지 반환
        empty_img = np.zeros((320, 320 * len(py_preds) if 'py_preds' in locals() else 320, 3), dtype=np.uint8)
        empty_heatmap = np.zeros((320, 320 * 2, 3), dtype=np.uint8)
        return empty_img, empty_heatmap