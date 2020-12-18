# OpenCV와 TensorFlow를 활용한 마스크 미착용자를 감지하는 자율주행 로봇

## 프로젝트 소개
- 2020-2 KNU 종합설계프로젝트2 팀 bdd의 프로젝트
- 실내 COVID-19 방역 강화를 위해 마스크 미착용자를 빠르게 발견할 필요가 있다. 
- 기존의 방식은 사람이 직접 확인해야 하는 불편함이 있고, 매 순간 확인하지 못하는 문제점이 있다. 
- 따라서 팀 bdd는 자율주행 터틀봇을 활용한 안전하고 효과적인 실내 관리 시스템을 제안한다.  

## 수행 기간
- 2020.09.03.∼12.16. (4개월)

## 참여자
  1. 터틀봇 자율주행  
      - gruR
  2. OpenCV와 TensorFlow를 활용한 마스크 착용 여부 판단  
      - yoo-jaein
      - kwonminsang

## 개발환경
- TurtleBot3 Burger (Ubuntu Mate 16.04)  
- Remote PC (Ubuntu Mate 16.04)  

## 시연영상 및 결과보고서
- https://github.com/yoo-jaein/Mask-Detecting-Turtlebot/tree/master/Document 참고

## 수행결과
1. 터틀봇 자율주행
- 장애물을 피해 주행하기위한 알고리즘 순서도를 만들고 그에 따라 개발했다.
- 초기 속도와 회전 각도는 환경에 따라 프로그래머가 설정을 바꿀 수 있다.
- 장애물을 탐지하는 범위와 후에 회전하는 각도 또한 조절할 수 있다.

2. OpenCV와 TensorFlow를 활용한 마스크 착용 여부 판단
- 얼굴 인식 모델은 caffemodel, 마스크 인식 모델은 tensorflow를 이용했다.
- 터틀봇의 카메라 영상을 Remote PC에서 받아와 마스크 착용자와 미착용자를 구분하는데 성공했다.

3. 마스크 미착용자에게 주의사항 전달
- TTS 출력을 위해 스피커를 납땜했다.

## 참고자료
- https://github.com/yoo-jaein/Mask-Detecting-Turtlebot/wiki 참고
