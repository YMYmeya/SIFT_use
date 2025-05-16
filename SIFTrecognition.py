import cv2

def sift_matching(query_img, target_img):
    # 创建SIFT对象
    sift = cv2.SIFT_create()

    # 检测特征点和计算特征描述符
    keypoints_query, descriptors_query = sift.detectAndCompute(query_img, None)
    keypoints_target, descriptors_target = sift.detectAndCompute(target_img, None)

    # 创建FLANN匹配器
    flann = cv2.FlannBasedMatcher()

    # 使用knnMatch进行特征点匹配
    matches = flann.knnMatch(descriptors_query, descriptors_target, k=2)

    # 进行 Lowe's ratio test 进行筛选
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 如果匹配对数超过一定阈值，则认为目标被识别
    if len(good_matches) > match_threshold:
        return True
    else:
        return False


# 读取查询图像和目标图像
query_img = cv2.imread('angle1.jpg', 0)  # 灰度图像
target_img = cv2.imread('angle2.jpg', 0)  # 灰度图像

# 设置匹配阈值
match_threshold = 10

# 使用SIFT进行目标识别
is_match = sift_matching(query_img, target_img)

# 打印结果
if is_match:
    print("目标被识别！")
else:
    print("目标未被识别！")
