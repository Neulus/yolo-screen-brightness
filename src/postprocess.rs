use opencv::core::{min_max_loc, no_array, Point, Range, Rect, Vector};
use opencv::core::{Mat_AUTO_STEP, CV_32F};
use opencv::dnn::nms_boxes;
use opencv::prelude::*;
use std::os::raw::c_void;

#[derive(Debug)]
pub struct BoxDetection {
    pub xmin: i32,
    pub ymin: i32,
    pub xmax: i32,
    pub ymax: i32,

    pub class: i32,
    pub conf: f32,
}

pub struct MatInfo {
    pub width: f32,
    pub height: f32,
}

pub struct Detections {
    pub detections: Vec<BoxDetection>,
}

pub fn post_process(
    outs: &Vector<Mat>,
    mat_info: &MatInfo,
    conf_thresh: f32,
    nms_thresh: f32,
) -> opencv::Result<Detections> {
    let mut det = outs.get(0)?;

    let rows = det.mat_size().get(1).unwrap();
    let cols = det.mat_size().get(2).unwrap();

    let mut boxes: Vector<opencv::core::Rect> = Vector::default();
    let mut scores: Vector<f32> = Vector::default();

    let mut indices: Vector<i32> = Vector::default();

    let mut class_index_list: Vector<i32> = Vector::default();

    let x_factor = mat_info.width / 640.0;
    let y_factor = mat_info.height / 640.0;

    unsafe {
        let mut min_val = Some(0f64);
        let mut max_val = Some(0f64);

        let mut min_loc = Some(Point::default());
        let mut max_loc = Some(Point::default());
        let mut mask = no_array();

        let data = det.ptr_mut(0)?.cast::<c_void>();

        // safe alternative needed..
        let m = Mat::new_rows_cols_with_data(rows, cols, CV_32F, data, Mat_AUTO_STEP)?;

        for r in 0..m.rows() {
            let cx: &f32 = m.at_2d::<f32>(r, 0)?;
            let cy: &f32 = m.at_2d::<f32>(r, 1)?;
            let w: &f32 = m.at_2d::<f32>(r, 2)?;
            let h: &f32 = m.at_2d::<f32>(r, 3)?;
            let sc: &f32 = m.at_2d::<f32>(r, 4)?;

            let score = *sc as f64;

            if score < conf_thresh.into() {
                continue;
            }
            let confs = m.row(r)?.col_range(&Range::new(5, m.row(r)?.cols())?)?;

            let c = (confs * score).into_result()?.to_mat()?;

            // find predicted class with highest confidence
            min_max_loc(
                &c,
                min_val.as_mut(),
                max_val.as_mut(),
                min_loc.as_mut(),
                max_loc.as_mut(),
                &mut mask,
            )?;

            scores.push(max_val.unwrap() as f32);
            class_index_list.push(max_loc.unwrap().x);

            boxes.push(Rect {
                x: (((*cx) - (*w) / 2.0) * x_factor).round() as i32,
                y: (((*cy) - (*h) / 2.0) * y_factor).round() as i32,
                width: (*w * x_factor).round() as i32,
                height: (*h * y_factor).round() as i32,
            });
            indices.push(r);
        }
    }

    nms_boxes(
        &boxes,
        &scores,
        conf_thresh,
        nms_thresh,
        &mut indices,
        1.0,
        0,
    )?;

    let mut final_boxes: Vec<BoxDetection> = Vec::default();

    for i in &indices {
        let indx = i as usize;

        let class = class_index_list.get(indx)?;

        let rect = boxes.get(indx)?;

        let bbox = BoxDetection {
            xmin: rect.x,
            ymin: rect.y,
            xmax: rect.x + rect.width,
            ymax: rect.y + rect.height,
            conf: scores.get(indx)?,
            class: class,
        };

        final_boxes.push(bbox);
    }

    Ok(Detections {
        detections: final_boxes,
    })
}
