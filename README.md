
# [TCSVT 2024] [SDPL: Shifting-Dense Partition Learning for UAV-view Geo-localization](https://ieeexplore.ieee.org/document/10587023)



## Dataset & Preparation
Download [University-1652](https://github.com/layumi/University1652-Baseline) upon request. You may use the request [template](https://github.com/layumi/University1652-Baseline/blob/master/Request.md).


## Evaluation University-1652
```
1.  Download the pre-trained weight: net_350.pth
(https://drive.google.com/file/d/1l2Dh7BtwafSZ9abPWuJ9maov2cCYUS8U/view?usp=sharing)

2.  Move the weight to ./model/pretrained/

3.  setting pad_value==0 in runt.sh

4.  sh test.sh
```

### Manually offset query images
```
If you want to adjust padding pixels, change pad_value of runt.sh.

If you want to adjust padding patterns, change line 115 of image_folder.py.

    For (+P,0), img = transforms.functional.pad(img,(self.pad,0,0,0),padding_mode='reflect')
    
    For (+P,+P), img = transforms.functional.pad(img,(self.pad,self.pad,0,0),padding_mode='reflect')
    
    For (-P,-P), img = transforms.functional.pad(img,(0,0,self.pad,self.pad),padding_mode='reflect')
    
    For (+P,-P), img = transforms.functional.pad(img,(self.pad,0,0,self.pad),padding_mode='reflect')
    
    For (-P,+P), img = transforms.functional.pad(img,(0,self.pad,self.pad,0),padding_mode='reflect')
```



## Our Related Works
```bibtex
@article{wang2024multiple,
  title={Multiple-environment Self-adaptive Network for Aerial-view Geo-localization},
  author={Wang, Tingyu and Zheng, Zhedong and Sun, Yaoqi and Yan, Chenggang and Yang, Yi and Chua, Tat-Seng},
  journal={Pattern Recognition},
  volume={152},
  pages={110363},
  year={2024},
  publisher={Elsevier}
}
```
```bibtex
@inproceedings{chen2023cross,
  title={A Cross-View Matching Method Based on Dense Partition Strategy for UAV Geolocalization},
  author={Chen, Yireng and Yang, Zihao and Chen, Quan},
  booktitle={Proceedings of the 2023 Workshop on UAVs in Multimedia: Capturing the World from a New Perspective},
  pages={19--23},
  year={2023}
}
```
```bibtex
@inproceedings{li2023drone,
  title={Drone Satellite Matching based on Multi-scale Local Pattern Network},
  author={Li, Haoran and Chen, Quan and Yang, Zhiwen and Yin, Jiong},
  booktitle={Proceedings of the 2023 Workshop on UAVs in Multimedia: Capturing the World from a New Perspective},
  pages={51--55},
  year={2023}
}
```


## Acknowledgement
The codes are based on LPN and FSRA. Please consider citing them.
```bibtex
@ARTICLE{wang2021LPN,
  title={Each Part Matters: Local Patterns Facilitate Cross-View Geo-Localization}, 
  author={Wang, Tingyu and Zheng, Zhedong and Yan, Chenggang and Zhang, Jiyong and Sun, Yaoqi and Zheng, Bolun and Yang, Yi},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  year={2022},
  volume={32},
  number={2},
  pages={867-879},
  doi={10.1109/TCSVT.2021.3061265}}
```
```bibtex
@ARTICLE{9648201,
  author={Dai, Ming and Hu, Jianhong and Zhuang, Jiedong and Zheng, Enhui},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={A Transformer-Based Feature Segmentation and Region Alignment Method for UAV-View Geo-Localization}, 
  year={2022},
  volume={32},
  number={7},
  pages={4376-4389},
  keywords={Transformers;Heating systems;Feature extraction;Drones;Satellites;Task analysis;Location awareness;Image retrieval;geo-localization;transformer;drone},
  doi={10.1109/TCSVT.2021.3135013}}
```
