import numpy as np


class Box:
    def __init__(self, p1, p2, conf):
        self.p1 = p1
        self.p2 = p2
        self.conf = conf

    def area(self):
        x1 = self.p1[0]
        y1 = self.p1[1]
        x2 = self.p2[0]
        y2 = self.p2[1]

        # safe check
        if x1 < x2 and y1 < y2:
            return (x2-x1)*(y2-y1)
        else:
            raise Exception("p1, p2 size comparsion violation")


class BoxManager:
    def __init__(self, threshold=0.5, gridcellsize=32):
        self.threshold = threshold
        self.gridcellsize = gridcellsize  # the default value 32 comes from 416/13

    def get_rect_points(self, cx, cy, w, h):
        p1 = (int(cx-w/2),  int(cy-h/2))
        p2 = (int(cx+w/2), int(cy + h/2))
        return p1, p2

    def convert_for_batch(self, pred_out_cxy, pred_out_rwh, pred_out_conf, applyNMS=False):

        boxlist = []

        batch_size = pred_out_cxy.shape[0]

        for i in range(batch_size):
            single_pred_out_cxy = pred_out_cxy[i, :, :, :]
            single_pred_out_rwh = pred_out_rwh[i, :, :, :]
            single_pred_out_conf = pred_out_conf[i, :, :, :]

            print("single_pred_out_cxy shape=", single_pred_out_cxy.shape)

            single_boxlist = self.convert_for_single_image(
                single_pred_out_cxy, single_pred_out_rwh, single_pred_out_conf, applyNMS=applyNMS)
            boxlist.append(single_boxlist)

        return boxlist

    def convert_for_single_image(self, pred_out_cxy, pred_out_rwh, pred_out_conf, applyNMS=False):

        boxlist = []

        print("pred_out_cxy shape=", pred_out_cxy.shape)

        index_arrs = np.where(pred_out_conf > self.threshold)

        grid_arr = index_arrs[0]
        box_arr = index_arrs[1]

        grid_arr = grid_arr.reshape(1, grid_arr.shape[0])
        box_arr = box_arr.reshape(1, box_arr.shape[0])

        assert grid_arr.shape[1] == box_arr.shape[1]

        for i in range(grid_arr.shape[1]):

            grid_index = grid_arr[0][i]
            box_index = box_arr[0][i]

            # print('grid_index = ', grid_index)
            # print('box_index = ', box_index)

            cx = pred_out_cxy[grid_index, box_index, 0]
            cy = pred_out_cxy[grid_index, box_index, 1]

            rw = pred_out_rwh[grid_index, box_index, 0]
            rh = pred_out_rwh[grid_index, box_index, 1]

            grid_x_index = grid_index % 13
            grid_y_index = int(grid_index / 13)

            o_x = grid_x_index * self.gridcellsize
            o_y = grid_y_index * self.gridcellsize

            abs_cx = o_x + cx * self.gridcellsize
            abs_cy = o_y + cy * self.gridcellsize

            abs_w = self.gridcellsize * rw
            abs_h = self.gridcellsize * rh

            p1, p2 = self.get_rect_points(abs_cx, abs_cy, abs_w, abs_h)

            box = Box(p1, p2, pred_out_conf[grid_index, box_index, 0])

            boxlist.append(box)

        if applyNMS:
            return self.applyNMS(boxlist)
        else:
            return boxlist

    def applyNMS(self, boxlist, iou_threshold=0.5):

        def find_box_with_max_conf(boxlist):
            maxbox = None
            temp_conf = 0.0
            for box in boxlist:
                if box.conf > temp_conf:
                    temp_conf = box.conf
                    maxbox = box
            return maxbox

        def calculate_iou(box1, box2):
            p1_1 = box1.p1
            p1_2 = box2.p1

            i1_x = max(p1_1[0], p1_2[0])
            i1_y = max(p1_1[1], p1_2[1])

            p2_1 = box1.p2
            p2_2 = box2.p2

            i2_x = min(p2_1[0], p2_2[0])
            i2_y = min(p2_1[1], p2_2[1])

            # just checking

            if i1_x < i2_x and i1_y < i2_y:
                # this is the only condition when iou is valid
                intersection = (i2_x - i1_x) * (i2_y - i1_y)
                box1_area = box1.area()
                box2_area = box2.area()
                union = box1_area + box2_area - intersection

                return intersection/union
            else:
                return 0.0

        def is_iou_over_threshold(box1, box2):
            calc_iou = calculate_iou(box1, box2)
            print("{} - {} iou={}".format(box1, box2, calc_iou))

            if calc_iou > self.iou_threshold:
                return True
            else:
                return False

        def merge_with_primebox(boxlist, primebox):
            boxlist.remove(primebox)
            print("boxlist after removing primebox({}) :".format(
                len(boxlist)), boxlist)

            if boxlist is None:
                print("boxlist is none returning None")
                return None

            boxes_to_remove = []
            for box in boxlist:
                if box is not primebox and is_iou_over_threshold(box, primebox):
                    print("removing box ", box)
                    boxes_to_remove.append(box)

            for box in boxes_to_remove:
                boxlist.remove(box)

            return boxlist

        def print_iou_among_boxes(primeboxlist):
            for i in range(len(primeboxlist)):
                for j in range(i+1, len(primeboxlist)):
                    box1 = primeboxlist[i]
                    box2 = primeboxlist[j]

                    print("{} - {} iou={}".format(box1,
                                                  box2, calculate_iou(box1, box2)))

        self.iou_threshold = iou_threshold

        if len(boxlist) == 0:
            return boxlist

        prime_box_list = []

        print("boxlist before entering merge loop:", boxlist)

        while boxlist is not None:
            primebox = find_box_with_max_conf(boxlist)
            if primebox is None:
                break
            prime_box_list.append(primebox)

            print("boxlist before calling merge_with_primebox, ", boxlist)

            boxlist = merge_with_primebox(boxlist, primebox)
            print("boxlist({})= ".format(len(boxlist)), boxlist)

        print("finished while loop")

        print("prime_box_list", prime_box_list)
        print_iou_among_boxes(prime_box_list)

        return prime_box_list

    def get_gt_boxes(self, gt_array):

        gtboxlist = []

        gt_conf = gt_array[:, :, 4]
        tt = np.where(gt_conf == 1.0)

        gt_boxes_size = len(tt[0])

        for i in range(gt_boxes_size):

            gt_conf_grid_index = tt[0][i]
            gt_conf_box_index = tt[1][i]

            gt_cx = gt_array[gt_conf_grid_index, gt_conf_box_index, 0]
            gt_cy = gt_array[gt_conf_grid_index, gt_conf_box_index, 1]
            gt_rw = gt_array[gt_conf_grid_index, gt_conf_box_index, 2]
            gt_rh = gt_array[gt_conf_grid_index, gt_conf_box_index, 3]

            grid_x_index = gt_conf_grid_index % 13
            grid_y_index = int(gt_conf_grid_index / 13)

            o_x = grid_x_index * self.gridcellsize
            o_y = grid_y_index * self.gridcellsize

            abs_cx = o_x + gt_cx * self.gridcellsize
            abs_cy = o_y + gt_cy * self.gridcellsize

            abs_w = self.gridcellsize * gt_rw
            abs_h = self.gridcellsize * gt_rh

            p1, p2 = self.get_rect_points(abs_cx, abs_cy, abs_w, abs_h)

            box = Box(p1, p2, 1.0)
            gtboxlist.append(box)

        return gtboxlist
