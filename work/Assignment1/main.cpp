#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include "Triangle.hpp"
#include "rasterizer.hpp"

constexpr double MY_PI = 3.1415926;

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos) {
  Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

  Eigen::Matrix4f translate;
  translate << 1, 0, 0, -eye_pos[0], 0, 1, 0, -eye_pos[1], 0, 0, 1,
    -eye_pos[2], 0, 0, 0, 1;

  view = translate * view;

  return view;
}

Eigen::Matrix4f get_rotation(Vector3f axis, float angle) {
  float a = angle * MY_PI / 180;

  // Rodrigues' Rotation Formula
  Eigen::Matrix3f rotateMat =
    cos(a) * Eigen::Matrix3f::Identity() +
    (1 - cos(a)) * (axis * axis.transpose()) +
    sin(a) * 
    (Eigen::Matrix3f() <<
      0        , -axis[2] , axis[1],
      axis[2]  , 0        , -axis[0],
      -axis[1] , axis[0]  , 0
    ).finished();

  return (Eigen::Matrix4f() <<
    rotateMat.block<1, 3>(0, 0), 0,
    rotateMat.block<1, 3>(1, 0), 0,
    rotateMat.block<1, 3>(2, 0), 0,
    0,         0,         0,     1
    ).finished();
}

Eigen::Matrix4f get_model_matrix(float rotation_angle, char whichAxis) {
  Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
  float a = rotation_angle * MY_PI / 180;

  // 现在这个函数可以根据 whichAxis 指示的 x y z 来获得在不同轴上的旋转变换矩阵
  // 且旋转中心是三角形的中心
  // 这样更有趣一些

  if (whichAxis == 'x') {
    model <<
      1, 0,      0,       0,
      0, cos(a), -sin(a), 0,
      0, sin(a), cos(a),  0,
      0, 0,      0,       1;
  }
  else if (whichAxis == 'y') {
    model <<
      cos(a),  0, sin(a), 0,
      0,       1, 0,      0,
      -sin(a), 0, cos(a), 0,
      0,       0, 0,      1;
  }
  else if (whichAxis == 'z') {
    model <<
      cos(a), -sin(a), 0, 0,
      sin(a), cos(a),  0, 0,
      0,      0,       1, 0,
      0,      0,       0, 1;
  }

  return
    (Eigen::Matrix4f() <<
      1, 0, 0, 0,
      0, 1, 0, 1,
      0, 0, 1, -2,
      0, 0, 0, 1).finished()
    * model *
    (Eigen::Matrix4f() <<
      1, 0, 0, 0,
      0, 1, 0, -1,
      0, 0, 1, 2,
      0, 0, 0, 1).finished();
}

Eigen::Matrix4f get_projection_matrix(float eye_fov,
  float aspect_ratio,
  float zNear,
  float zFar) {
  // Students will implement this function
  Eigen::Matrix4f projection = Eigen::Matrix4f::Identity(),
    orthoProjMat, persp2ortho;

  // [1, -1]^3 的所有参数
  float l, r, b, t, f = -zFar, n = -zNear;
  b = -tan(eye_fov * MY_PI / 180) * -n / 2;
  t = -b;
  l = -(t - b) * aspect_ratio / 2;
  r = -l;

  // 正交投影变换矩阵
  orthoProjMat =
    (Eigen::Matrix4f() <<
      2 / (r - l), 0,           0,           0,
      0,           2 / (t - b), 0,           0,
      0,           0,           2 / (n - f), 0,
      0,           0,           0,           1
      ).finished()
    *
    (Eigen::Matrix4f() <<
      1, 0, 0, -(r + l) / 2,
      0, 1, 0, -(t + b) / 2,
      0, 0, 1, -(n + f) / 2,
      0, 0, 0, 1
      ).finished()
    ;

  // 平面压缩变换矩阵
  persp2ortho <<
    n, 0, 0,     0,
    0, n, 0,     0,
    0, 0, n + f, -n * f,
    0, 0, 1,     0
    ;

  // 先压缩变换后正交投影
  projection = orthoProjMat * persp2ortho;
  return projection;
}

int main(int argc, const char** argv) {
  float angle = 0;
  bool command_line = false;
  std::string filename = "output.png";

  if (argc >= 3) {
    command_line = true;
    angle = std::stof(argv[2]);  // -r by default
    if (argc == 4) {
      filename = std::string(argv[3]);
    }
    else
      return 0;
  }

  rst::rasterizer r(700, 700);

  Eigen::Vector3f eye_pos = { 0, 0, 5 };

  std::vector<Eigen::Vector3f> pos{ {2, 0, -2}, {0, 2, -2}, {-2, 0, -2} };

  std::vector<Eigen::Vector3i> ind{ {0, 1, 2} };

  auto pos_id = r.load_positions(pos);
  auto ind_id = r.load_indices(ind);

  int key = 0;
  int frame_count = 0;

  if (command_line) {
    r.clear(rst::Buffers::Color | rst::Buffers::Depth);

    r.set_model(get_model_matrix(angle, 'z'));
    r.set_view(get_view_matrix(eye_pos));
    r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

    r.draw(pos_id, ind_id, rst::Primitive::Triangle);
    cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
    image.convertTo(image, CV_8UC3, 1.0f);

    cv::imwrite(filename, image);

    return 0;
  }

  // 这里稍有改动
  // 在外部作用域维护模型的变换状态，以实现在所有轴上进行旋转
  Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
  while (key != 27) {
    r.clear(rst::Buffers::Color | rst::Buffers::Depth);
    // 将维护的模型变换状态直接传进去，这个模型的变换在下面控制
    r.set_model(model);
    r.set_view(get_view_matrix(eye_pos));
    r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

    r.draw(pos_id, ind_id, rst::Primitive::Triangle);

    cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
    image.convertTo(image, CV_8UC3, 1.0f);
    cv::imshow("image", image);
    key = cv::waitKey(10);

    std::cout << "frame count: " << frame_count++ << '\n';

    // 这里负责更新模型的变换状态，而不是仅更新角度
    if (key == 'q') {
      model = get_model_matrix(10, 'y') * model;
    }
    else if (key == 'e') {
      model = get_model_matrix(-10, 'y') * model;
    }
    else if (key == 'w') {
      model = get_model_matrix(-10, 'x') * model;
    }
    else if (key == 's') {
      model = get_model_matrix(10, 'x') * model;
    }
    else if (key == 'a') {
      model = get_model_matrix(10, 'z') * model;
    }
    else if (key == 'd') {
      model = get_model_matrix(-10, 'z') * model;
    }
  }

  return 0;
}
