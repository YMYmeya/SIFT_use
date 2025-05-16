import cv2
import numpy as np

# 创建SIFT对象
sift = cv2.SIFT_create()

# 创建FLANN匹配器
flann = cv2.FlannBasedMatcher()

# 读取待追踪主体图像
init_frame = cv2.imread('hide_cover.jpg', 0)  # 灰度图像

# 检测初始帧特征点和计算特征描述符
keypoints_init, descriptors_init = sift.detectAndCompute(init_frame, None)

# 创建视频捕获对象
cap = cv2.VideoCapture('hide_cover.mp4')

# 初始目标位置
bbox = (0, 0, 0, 0)

while True:
    # 读取下一帧图像
    ret, frame = cap.read()

    if not ret:
        break

    # 转换为灰度图像
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测当前帧特征点和计算特征描述符
    keypoints_frame, descriptors_frame = sift.detectAndCompute(gray_frame, None)

    # 使用FLANN匹配器进行特征点匹配
    matches = flann.knnMatch(descriptors_init, descriptors_frame, k=2)

    # 筛选匹配对
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 获取匹配对中的特征点位置
    src_points = np.float32([keypoints_init[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_points = np.float32([keypoints_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 使用RANSAC算法估计变换矩阵
    if len(good_matches) >= 4:
        M, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

        # 计算目标位置（左上角坐标和宽高）
        h, w = init_frame.shape
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)

        if M is not None:
            dst = cv2.perspectiveTransform(pts, M)
            x, y, w, h = cv2.boundingRect(dst)
            # 更新目标位置
            bbox = (x, y, w, h)

    # 绘制目标框
    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 255, 0), 2)

    # 显示结果
    cv2.imshow("Tracking", frame)

    # 按下 ESC 键退出
    if cv2.waitKey(1) == 27:
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()

