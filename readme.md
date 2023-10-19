# Conditional Variatinal Auto Encoder

url : https://www.youtube.com/watch?v=rNh2CrTFpm4&t=2321s (오토인코더의 모든 것2, 이활석 박사)

## Result

* Conditional Variational Auto Encoder의 Latent vector 분포
  * 모든 class를 한번에 좌표계에 표시
    <p align="center">
      <img src="./result/figure_all_index.png" width="600" height="400" />
    </p>
  * 각각의 class를 따로 좌표계에 표시
    <p align="center">
      <img src="./result/figure_index_0to10.png" width="800" height="400" />
    </p>
각각의 클래스 모두가 normal distribution을 따름<br>

* y(class)를 고정시켜놓고 z를 변화시켜가며 확인
  * 순서대로 y가 0 ~ 9 일 때의 출력
    <p align="left">
      <img src="./result/z_map_0.jpg" width="250" height="250" />
      <img src="./result/z_map_1.jpg" width="250" height="250" />
      <img src="./result/z_map_2.jpg" width="250" height="250" />
    </p>
    <p align="left">
      <img src="./result/z_map_3.jpg" width="250" height="250" />
      <img src="./result/z_map_4.jpg" width="250" height="250" />
      <img src="./result/z_map_5.jpg" width="250" height="250" />
    </p>
    <p align="left">
      <img src="./result/z_map_6.jpg" width="250" height="250" />
      <img src="./result/z_map_7.jpg" width="250" height="250" />
      <img src="./result/z_map_8.jpg" width="250" height="250" />
    </p>
    <p align="left">
      <img src="./result/z_map_9.jpg" width="250" height="250" />
    </p>

z의 두 차원이 의미하는 특징을 확인할 수 있음.
