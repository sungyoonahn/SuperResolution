차량 번호판 데이터 생성

목적
- YOLO V4 를 사용하여 차량 번호판 추출

YOLO 설치
- https://github.com/AlexeyAB/darknet 에서 윈도우(vcpkg 방식 권장), 리눅스 환경에 맞는 설치 방식에 따라 환경 설치
- custum YOLO 파일 다운로드 (darknet)

데이터 생성
- 데이터 파일(images) 속 python check.py 실행 -> output.txt 생성
- output.txt를 darknet 파일로 이동
- darknet에서 darknet.exe detector test data/obj.data cfg/yolo-obj.cfg LP.weights -ext_output -dont_show -out output.json < output.txt  실행
- output.json 파일 생성
- output.json 속 "\" 를 "\\" 로 변환(윈도우 파일 인식 에러)
- darknet에서 python img_crop_from_json.py 실행
- 이미지 경로와 동일한 경로에 있는 images_cropped폴더안에 차량 번호판 데이터 저장

-------

Super_Resolution 학습
- 생성된 데이터를 SuperResolution\\LP 로 이동
- Super_Resolution에서 python dataset_csv.py 실행
- SuperResolution\\LP에 list_eval_partition.csv생성(train, val, test 데이터 분리 csv)
- Super_Resolution에서 python preprocess.py 실행 (SuperResolution\\LP\\processed 속에 x_train, x_val, x_test, y_train, y_val, y_test 파일 생성 필요)
- Super_Resolution에서 python train.py 실행
- Super_Resolution\\models에 model_LPR.h5 파일 생성

Super_Resolution 결과 확인
- Super_Resolution에서 python gui.py 실행
- test 데이터 수보다 적은 양의 정수 입력 후 check 버튼 클릭
- 원본 이미지 Super_Resoluion전 이미지 후 이미지 확인 가능(psnr 값 포함)