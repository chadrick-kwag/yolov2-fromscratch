import numpy as np
import cv2


class PredictionProcessor_v1:

    threshold = 0.5

    def get_rect_points(self, cx, cy, w, h):
        p1 = (int(cx-w/2),  int(cy-h/2))
        p2 = (int(cx+w/2), int(cy + h/2))
        return p1, p2

    def draw_all_bbx(self, pred_out_cxy=None, pred_out_rwh=None, pred_out_conf=None, gt_arr=None, image=None):
        # image should be given in 416x416 size

        index_arrs = np.where(pred_out_conf > self.threshold)

        print("over threshold conf count=", len(index_arrs[0]))

        grid_arr = index_arrs[0]
        box_arr = index_arrs[1]

        grid_arr = grid_arr.reshape(1, grid_arr.shape[0])
        box_arr = box_arr.reshape(1, box_arr.shape[0])

        assert grid_arr.shape[1] == box_arr.shape[1]

        debug = False

        # reset cv2 image
        reduced_image = image.copy()

        for i in range(grid_arr.shape[1]):

            grid_index = grid_arr[0][i]
            box_index = box_arr[0][i]

            # print('grid_index = ', grid_index)
            # print('box_index = ', box_index)

            cx = pred_out_cxy[grid_index, box_index, 0]
            cy = pred_out_cxy[grid_index, box_index, 1]

            rw = pred_out_rwh[grid_index, box_index, 0]
            rh = pred_out_rwh[grid_index, box_index, 1]

            grid_cell_size = 416/13

            grid_x_index = grid_index % 13
            grid_y_index = int(grid_index / 13)

            o_x = grid_x_index * grid_cell_size
            o_y = grid_y_index * grid_cell_size

            abs_cx = o_x + cx * grid_cell_size
            abs_cy = o_y + cy * grid_cell_size

            abs_w = grid_cell_size * rw
            abs_h = grid_cell_size * rh

            p1, p2 = self.get_rect_points(abs_cx, abs_cy, abs_w, abs_h)

            # print(p1,p2)

            cv2.rectangle(reduced_image, p1, p2, (0, 0, 255), thickness=2)

            if debug:
                break

        # draw gt box
        gt_conf = gt_arr[:, :, 4]
        # print("gt_conf shape",gt_conf.shape)
        tt = np.where(gt_conf == 1.0)
        # print(tt)
        gt_conf_grid_index = tt[0][0]
        gt_conf_box_index = tt[1][0]

        gt_cx = gt_arr[gt_conf_grid_index, gt_conf_box_index, 0]
        gt_cy = gt_arr[gt_conf_grid_index, gt_conf_box_index, 1]
        gt_rw = gt_arr[gt_conf_grid_index, gt_conf_box_index, 2]
        gt_rh = gt_arr[gt_conf_grid_index, gt_conf_box_index, 3]

        grid_cell_size = 416/13

        grid_x_index = gt_conf_grid_index % 13
        grid_y_index = int(gt_conf_grid_index / 13)

        o_x = grid_x_index * grid_cell_size
        o_y = grid_y_index * grid_cell_size

        abs_cx = o_x + gt_cx * grid_cell_size
        abs_cy = o_y + gt_cy * grid_cell_size

        abs_w = grid_cell_size * gt_rw
        abs_h = grid_cell_size * gt_rh

        p1, p2 = self.get_rect_points(abs_cx, abs_cy, abs_w, abs_h)

        # print(p1,p2)

        cv2.rectangle(reduced_image, p1, p2, (0, 255, 0), thickness=1)

        return_image = cv2.cvtColor(reduced_image, cv2.COLOR_RGB2BGR)

        return return_image

    # def extract_all_boxes(self,pred_out_cxy=None, pred_out_rwh=None, pred_out_conf=None):

    def draw_all_boxes(self, boxlist, image, gtboxlist=None):
        reduced_image = image.copy()

        for box in boxlist:
            cv2.rectangle(reduced_image, box.p1, box.p2,
                          (0, 0, 255), thickness=2)

        if gtboxlist is not None:
            for box in gtboxlist:
                cv2.rectangle(reduced_image, box.p1, box.p2,
                              (0, 255, 0), thickness=1)

        # draw gt box

        return_image = cv2.cvtColor(reduced_image, cv2.COLOR_RGB2BGR)
        return return_image
