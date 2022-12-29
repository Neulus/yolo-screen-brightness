use brightness::blocking::{brightness_devices, Brightness, BrightnessDevice};
use opencv::dnn::read_net_from_onnx_buffer;
use opencv::prelude::*;
use opencv::{
    core::{Scalar, Vector},
    videoio::VideoCapture,
};
use postprocess::*;
use std::process::exit;
use std::thread::sleep;
use std::time::Instant;

mod postprocess;

fn main() {
    let network_file = include_bytes!("../yolov5n.onnx");
    let network_file = Vector::from_slice(network_file);

    let mut network = read_net_from_onnx_buffer(&network_file).unwrap_or_else(|e| {
        eprintln!("모델을 로딩하는데 실패하였습니다. {}", e);
        exit(1)
    });

    let mut video = VideoCapture::default().unwrap_or_else(|e| {
        eprintln!("카메라를 감지 할 수 없습니다. {}", e);
        exit(1)
    });

    video.open(0, 0).unwrap_or_else(|e| {
        eprintln!("카메라를 열 수 없습니다. {}", e);
        exit(1)
    });

    let mut last_human_detection = Instant::now();
    let mut is_power_saving = false;

    loop {
        sleep(std::time::Duration::from_millis(100));

        let mut frame = Mat::default();
        let retrived = video.read(&mut frame).unwrap_or_else(|e| {
            eprintln!("카메라에서 데이터를 읽어오는데 실패하였습니다. {}", e);
            exit(1)
        });

        if !retrived {
            break;
        }

        let mat_info = MatInfo {
            width: frame.cols() as f32,
            height: frame.rows() as f32,
        };

        let image = opencv::dnn::blob_from_image(
            &frame,
            1.0 / 255.0,
            opencv::core::Size_::new(640, 640),
            Scalar::new(0f64, 0f64, 0f64, 0f64),
            true,
            false,
            opencv::core::CV_32F,
        )
        .unwrap_or_else(|e| {
            eprintln!("이미지를 작게 변환하는데 실패하였습니다. {}", e);
            exit(1)
        });

        let mut outs: Vector<Mat> = Vector::default();

        network
            .set_input(&image, "", 1.0, Scalar::default())
            .unwrap_or_else(|e| {
                eprintln!("모델에 데이터를 입력하는데 실패하였습니다. {}", e);
                exit(1)
            });
        network
            .forward(
                &mut outs,
                &network.get_unconnected_out_layers_names().unwrap(),
            )
            .unwrap_or_else(|e| {
                eprintln!("모델을 실행하는데 실패하였습니다. {}", e);
                exit(1)
            });

        let detections = post_process(&outs, &mat_info, 0.1, 0.1).unwrap_or_else(|e| {
            eprintln!("결과를 처리하는데 실패하였습니다. {}", e);
            exit(1)
        });

        let is_human = detections
            .detections
            .iter()
            .any(|detection| detection.class == 0);

        if is_human {
            if is_power_saving {
                println!("밝기를 올립니다.");
                for device in brightness_devices() {
                    device.unwrap().set(100).unwrap();
                }
                is_power_saving = false;
            }
            last_human_detection = Instant::now();
        } else {
            let elapsed = last_human_detection.elapsed();
            if elapsed.as_secs() > 5 {
                if !is_power_saving {
                    println!("밝기를 내립니다.");
                    for device in brightness_devices() {
                        device.unwrap().set(0).unwrap();
                    }
                    is_power_saving = true;
                }
            }
        }
    }
}
