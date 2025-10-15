import numpy as np
import scipy.linalg
import math
import threading
import time
import requests
import serial
from AHRS.xdaSerialHandler import SerialHandler
from AHRS.XbusPacket import XbusPacket
from AHRS.xdaDataPacketParser import DataPacketParser, XsDataPacket
import keyboard
import csv
import os
from datetime import datetime
from globalpath_v4 import calculate_route_json
from pyproj import Proj, Transformer
from ouster.position_distance import ouster_distance

gps_data = {"Heading": None, "Lat": None, "Lon": None}
attitude_data = {"Roll": None, "Pitch": None, "Yaw": None}
last_update_time = {"gps": 0, "ahrs": 0}
ouster_front = None
ouster_event = threading.Event()


def collect_ouster_data():
    global ouster_front
    while True:
        ouster_event.wait()
        try:
            distance = ouster_distance(sensor_ip="192.168.50.131", pos="front")
            ouster_front = distance if distance is not None else None
        except Exception as e:
            print(f"Ouster 資料讀取失敗: {e}")
            ouster_front = None
        time.sleep(0.1)


ouster_thread = threading.Thread(target=collect_ouster_data, daemon=True)
ouster_thread.start()

file_timestamp = datetime.strftime(datetime.now(), "%Y_%m_%d_%H_%M_%S")
file_name = f"{file_timestamp}_multi_docking.csv"
file_path = f"./data/{file_name}"

with open(file_path, "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(
        [
            "Timestamp",
            "Goal_point",
            "Heading",
            "Lat",
            "Lon",
            "Roll",
            "Pitch",
            "Yaw",
            "target_lat",
            "target_lon",
            "yaw_error",
            "rudder_calc",
            "distance",
            "ouster_front",
        ]
    )


class MultiDimKalmanFilter:
    def __init__(
        self,
        dt,
        process_noise=1e-7,
        measurement_noise=2e-5,
        estimate_error=1e1,
        init_pos=None,
    ):
        """
        初始化卡爾曼濾波器
        :param dt: 兩次測量之間的時間間隔
        :param process_noise: 過程噪聲
        :param measurement_noise: 測量噪聲
        :param estimate_error: 初始估計誤差
        :param init_pos: 初始位置 (lat, lon)
        """
        self.dt = dt

        # 狀態轉移矩陣 (F) - 6 維狀態空間 [lat, lon, v_lat, v_lon, a_lat, a_lon]
        self.F = np.array(
            [
                [1, 0, dt, 0, 0.5 * dt**2, 0],
                [0, 1, 0, dt, 0, 0.5 * dt**2],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        # 觀測矩陣 (H) - 只觀測 [lat, lon]
        self.H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])

        # 初始誤差協方差矩陣 (P)
        self.P = np.eye(6) * estimate_error

        # 過程噪聲協方差矩陣 (Q) - 自適應調整
        self.process_noise_scale = process_noise
        self.Q = np.eye(6) * process_noise

        # 測量噪聲協方差矩陣 (R) - 自適應調整
        self.R = np.array([[measurement_noise, 0], [0, measurement_noise]])

        # 初始狀態向量 [lat, lon, v_lat, v_lon, a_lat, a_lon]
        self.x = np.zeros(6)
        if init_pos is not None:
            self.x[:2] = init_pos

        # 建立經緯度 <-> UTM 轉換器
        self.transformer = Transformer.from_crs(
            "EPSG:4326", "EPSG:3857"
        )  # WGS84 (經緯度) 轉 UTM
        self.transformer_inverse = Transformer.from_crs(
            "EPSG:3857", "EPSG:4326"
        )  # UTM 轉回 WGS84

    def update(self, measurement):
        """
        更新卡爾曼濾波器
        :param measurement: 當前 GPS 測量值 (lat, lon)
        :return: 濾波後的 (lat, lon)
        """
        # 轉換經緯度到 UTM
        utm_lat, utm_lon = self.transformer.transform(measurement[0], measurement[1])
        measurement_utm = np.array([utm_lat, utm_lon])

        # 計算測量變化量 (自適應 R)
        delta_lat, delta_lon = 0, 0
        if hasattr(self, "prev_measurement"):
            delta_lat = abs(measurement[0] - self.prev_measurement[0])
            delta_lon = abs(measurement[1] - self.prev_measurement[1])

        self.R = np.array(
            [[max(3e-7, delta_lat * 2e-7), 0], [0, max(3e-7, delta_lon * 2e-7)]]
        )
        self.prev_measurement = measurement  # 記錄上一個測量值

        # 計算過程噪聲 Q
        process_noise_scale = max(5e-6, min(delta_lat * 5e-2, delta_lon * 5e-2))
        self.Q = np.eye(6) * process_noise_scale

        # 狀態預測
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        # 計算誤差
        y = measurement_utm - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # 更新狀態
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)

        # 轉回 WGS84 (經緯度)
        filtered_lat, filtered_lon = self.transformer_inverse.transform(
            self.x[0], self.x[1]
        )
        return filtered_lat, filtered_lon


class DataCollector:
    # 當資料可用時的回調函數
    def on_live_data_available(packet):
        xbus_data = XsDataPacket()
        DataPacketParser.parse_data_packet(packet, xbus_data)

        if xbus_data.eulerAvailable:
            attitude_data["Roll"] = xbus_data.euler[0]
            attitude_data["Pitch"] = xbus_data.euler[1]
            attitude_data["Yaw"] = xbus_data.euler[2]
            # print(f"AHRS 資料更新: Roll={attitude_data['Roll']}, Pitch={attitude_data['Pitch']}, Yaw={attitude_data['Yaw']}")
            last_update_time["ahrs"] = time.time()  # 更新資料時間

    # 持續收集 GPS 資料的函數
    def collect_gps_data_KalmanFilter(gps_ser):
        global gps_data
        gps_list = []
        newtime = 0
        pretime = 0
        while True:
            try:
                gps_result = gps_ser.readline().decode("ascii", errors="ignore").strip()
                if gps_result.startswith("$IIGLL") or gps_result.startswith("$IIHDG"):
                    if gps_result.startswith("$IIGLL"):  # 處理 GLL 資料
                        parts = gps_result.split(",")
                        if len(parts) >= 6:
                            newtime = time.time()
                            lat_deg = float(parts[1][:2]) + float(parts[1][2:]) / 60
                            lon_deg = float(parts[3][:3]) + float(parts[3][3:]) / 60
                            lat_dir = parts[2]
                            lon_dir = parts[4]

                            if lat_dir == "S":
                                lat_deg = -lat_deg
                            if lon_dir == "W":
                                lon_deg = -lon_deg

                            gps_list.append((lat_deg, lon_deg))
                            # print(f"o_lat:{lat_deg},o_lon:{lon_deg}")
                            if len(gps_list) == 2:
                                timestamp = newtime - pretime
                                pretime = newtime

                                kf = MultiDimKalmanFilter(
                                    timestamp,
                                    process_noise=1e-2,
                                    measurement_noise=1e-2,
                                    estimate_error=1e0,
                                )
                                filtered_positions = []

                                for lat, lon in gps_list:
                                    measurement = np.array([lat, lon])
                                    filtered_lat_lon = kf.update(measurement)
                                    filtered_positions.append(filtered_lat_lon)

                                # 輸出濾波後的結果
                                for i, (lat, lon) in enumerate(filtered_positions):
                                    gps_data["Lat"] = lat_deg
                                    gps_data["Lon"] = lon_deg
                                    # print(f"GPS GLL 資料處理完成: 緯度={lat_deg}, 經度={lon_deg}")
                                gps_list = []

                    elif gps_result.startswith("$IIHDG"):  # 處理 HDG 資料
                        parts = gps_result.split(",")
                        if len(parts) >= 2:
                            heading = float(parts[1])
                            gps_data["Heading"] = heading
                            # print(f"GPS HDG 資料處理完成: 航向={heading}")

                    last_update_time["gps"] = time.time()  # 更新資料時間

            except ValueError as e:
                print(f"GPS 資料解析失敗: {e}")
            except Exception as e:
                print(f"GPS 讀取失敗: {e}")
            # time.sleep(0.05)  # 減少忙碌等待的 CPU 佔用

    # 持續收集 AHRS 資料的函數
    def collect_ahrs_data(serial_handler, packet):
        while True:
            try:
                byte = serial_handler.read_byte()
                packet.feed_byte(byte)
                last_update_time["ahrs"] = time.time()  # 更新資料時間
            except RuntimeError:
                continue
            except Exception as e:
                print(f"AHRS 讀取失敗: {e}")
            # time.sleep(0.05)  # 減少忙碌等待的 CPU 佔用

    gps_ser = None  # 全域變數，確保可以存取
    serial_handler = None  # 全域變數
    packet = None  # 全域變數
    gps_thread = None  # 新增全域變數
    ahrs_thread = None  # 新增全域變數

    @staticmethod
    def DataCollector(
        gps_port="COM7",
        gps_baudrate=9600,
        ahrs_port="COM12",
        ahrs_baudrate=115200,
        switch="open",
    ):
        global gps_ser, serial_handler, packet, gps_thread, ahrs_thread  # 讓這些變數可以跨函數存取

        if switch == "open":
            try:
                gps_ser = serial.Serial(gps_port, gps_baudrate, timeout=1)
            except Exception as e:
                print(f"GPS 串列埠打開失敗: {e}")
                return

            try:
                serial_handler = SerialHandler(ahrs_port, ahrs_baudrate)
                packet = XbusPacket(
                    on_data_available=lambda p: DataCollector.on_live_data_available(p)
                )

                go_to_config = bytes.fromhex("FA FF 30 00")
                go_to_measurement = bytes.fromhex("FA FF 10 00")

                serial_handler.send_with_checksum(go_to_config)
                serial_handler.send_with_checksum(go_to_measurement)
            except Exception as e:
                print(f"AHRS 串列埠打開失敗: {e}")
                return

            try:
                # 啟動 GPS 和 AHRS 的資料收集執行緒
                gps_thread = threading.Thread(
                    target=DataCollector.collect_gps_data_KalmanFilter,
                    args=(gps_ser,),
                    daemon=True,
                )
                ahrs_thread = threading.Thread(
                    target=DataCollector.collect_ahrs_data,
                    args=(serial_handler, packet),
                    daemon=True,
                )

                gps_thread.start()
                ahrs_thread.start()
            except Exception as e:
                print(f"線程開啟錯誤: {e}")
                return

        elif switch == "close":
            try:
                print("正在關閉 GPS 和 AHRS 執行緒...")

                # 確保線程變數已定義
                if gps_thread is not None:
                    gps_thread.join(timeout=1)
                else:
                    print("gps_thread 未初始化，無需關閉")

                if ahrs_thread is not None:
                    ahrs_thread.join(timeout=1)
                else:
                    print("ahrs_thread 未初始化，無需關閉")

                # 確保 GPS 串列埠已開啟才關閉
                if gps_ser is not None and gps_ser.is_open:
                    gps_ser.close()
                    print("GPS 串列埠已關閉")
                else:
                    print("gps_ser 未開啟，無需關閉")

                if serial_handler is not None and serial_handler.serial_port.is_open:
                    serial_handler.serial_port.close()
                    print("AHRS 串列埠已關閉")
                else:
                    print("serial_handler 未開啟，無需關閉")

            except Exception as e:
                print(f"關閉過程中發生錯誤: {e}")


class Cal_Rudder_Engine:
    # 計算目標航向角
    def calculate_target_heading(lat, lon, lat_ref, lon_ref):
        # 建立 UTM 投影轉換器（使用 WGS84 橢球體）
        proj = Proj(proj="utm", zone=51, ellps="WGS84", datum="WGS84")

        # 轉換經緯度為 UTM 坐標
        x1, y1 = proj(lon, lat)
        x2, y2 = proj(lon_ref, lat_ref)

        # 計算 UTM 平面距離（歐幾里得距離）
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # **修正 UTM 方位角計算**
        theta = math.atan2(x2 - x1, y2 - y1)  # 修正 X/Y 順序
        azimuth = (math.degrees(theta) + 360) % 360
        # 確保結果在 0 到 360 度之間
        target_heading = (azimuth + 360) % 360

        return target_heading

    # 計算角度差
    def calculate_heading_difference(current_heading, target_heading):
        # 計算角度差，並確保在 -180 到 180 之間
        difference = (target_heading - current_heading + 180) % 360 - 180
        return difference

    # 計算 UTM 投影距離
    def calculate_distance(lat1, lon1, lat2, lon2):
        # 建立 UTM 投影轉換器（使用 WGS84 橢球體）
        proj = Proj(proj="utm", zone=51, ellps="WGS84", datum="WGS84")

        # 轉換經緯度為 UTM 坐標
        x1, y1 = proj(lon1, lat1)
        x2, y2 = proj(lon2, lat2)

        # 計算 UTM 平面距離（歐幾里得距離）
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        return distance

    # 取參考點
    def get_next_target(current_lat, current_lon, goal_point):

        if goal_point:
            if (
                Cal_Rudder_Engine.calculate_distance(
                    current_lat, current_lon, goal_point[0], goal_point[1]
                )
                <= 5.4
            ):
                return None, None
            # if Cal_Rudder_Engine.calculate_distance(current_lat,current_lon,goal_point[0],goal_point[1])<=5:
            #     return None,None
            try:
                target = calculate_route_json(
                    [current_lat, current_lon], [goal_point[0], goal_point[1]]
                )
            except:
                target = goal_point
            # target[0] = target[0]
            distance_to_final = Cal_Rudder_Engine.calculate_distance(
                target[0], target[1], goal_point[0], goal_point[1]
            )
            if distance_to_final >= 150:
                target[0], target[1] = target[0], target[1]
            else:
                target[0], target[1] = goal_point[0], goal_point[1]

            return target[0], target[1]

        else:

            return None, None

    # 舵角計算函數，增加返回舵角和計算的相關資訊
    def RudderAngleCalculation(ship_data, goal_point, yaw_rate):
        delta_value_limited = None
        distance = Cal_Rudder_Engine.calculate_distance(
            ship_data["Lat"], ship_data["Lon"], goal_point[0], goal_point[1]
        )

        # 定義系統矩陣
        A = np.array([[0.9521, 0.0479], [1, 0]])  # 狀態轉移矩陣 A
        B = np.array([[-0.2043], [0]])  # 輸入矩陣 B
        # 根據距離動態調整 Q 和 R（權重矩陣）
        if distance >= 350:
            Q = np.array([[50, 0], [0, 1]])  # 遠距離，強調yaw_error修正
            R = np.array([[15]])
        elif 350 > distance >= 200:
            Q = np.array([[20, 0], [0, 1]])  # 中距離，中等調整
            R = np.array([[20]])
        else:
            Q = np.array([[5, 0], [0, 1]])  # 近距離，平穩控制
            R = np.array([[27]])
        # # 調整權重矩陣
        # Q = np.array(
        #     [[5, 0], [0, 1]]  # 增加 yaw_error 的權重，減少過度響應
        # )  # 狀態權重矩陣 Q，控制誤差的重要性
        # R = np.array([[23]])  # 增加 R 的值，減少控制輸入的幅度

        # 求解離散型 Riccati 方程以得到矩陣 P
        P = scipy.linalg.solve_discrete_are(A, B, Q, R)

        # 計算最優增益矩陣 K
        K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

        # 獲取目標點
        target_point = Cal_Rudder_Engine.get_next_target(
            ship_data["Lat"], ship_data["Lon"], goal_point
        )
        if target_point[0] is not None and target_point[1] is not None:
            target_lat, target_lon = target_point
        else:
            return 0, 0, 0, 0

        # 計算目標航向和誤差
        target_heading = Cal_Rudder_Engine.calculate_target_heading(
            ship_data["Lat"], ship_data["Lon"], target_lat, target_lon
        )
        yaw_error = Cal_Rudder_Engine.calculate_heading_difference(
            ship_data["Heading"], target_heading
        )

        # 構建狀態向量
        X = np.array([[yaw_error], [yaw_rate]])

        # 計算控制輸出（舵角）
        delta = -K @ X

        # 使用 LQR 控制器計算舵角輸出
        delta = -K @ X
        delta_value = float(delta[0, 0])  # 轉為純量

        # 限制舵角範圍在 [-25, 25]
        delta_value_limited = int(max(min(delta_value, 25), -25))

        return delta_value_limited, target_lat, target_lon, yaw_error

    # 引擎計算
    def EngineCalculation(now_gps, target_point, final_point=False):
        distance = Cal_Rudder_Engine.calculate_distance(
            now_gps[0], now_gps[1], target_point[0], target_point[1]
        )
        if final_point == True:
            if distance >= 120:
                return 1.1
            elif 120 > distance >= 35:
                return 1
            elif 35 > distance > 0:
                return 0.6
            else:
                return 0
        elif final_point == False:
            if distance >= 250:
                return 1.8
            elif 250 > distance >= 150:
                return 1.5
            elif 150 > distance >= 100:
                return 1.3
            elif 100 > distance >= 80:
                return 1.3
            elif 80 > distance > 0:
                return 1
            else:
                return 0


# 傳送API
def DataTransmitter(command=None, command_data=None):
    command_dict = {
        "stop": {
            "Command": 666,
            "Left_Speed": 0,
            "Right_Speed": 0,
            "Left_Rudder": 0,
            "Right_Rudder": 0,
            "Range": 0.1,
        },
        "back": {
            "Command": 666,
            "Left_Speed": -0.8,
            "Right_Speed": -0.8,
            "Left_Rudder": 0,
            "Right_Rudder": 0,
            "Range": 0.1,
        },
        "left_back": {
            "Command": 666,
            "Left_Speed": -0.8,
            "Right_Speed": -0.8,
            "Left_Rudder": -20,
            "Right_Rudder": -20,
            "Range": 0.1,
        },
    }
    if command:
        command_data = command_dict[command]
    api_url = "http://127.0.0.1:5899/control"
    try:
        response = requests.post(api_url, json=command_data)
        if response.status_code == 200:
            print("=====================!next!=====================")
            return True
        else:
            print(f"指令發送失敗: {response.status_code}")
    except requests.RequestException as e:
        print(f"API 請求失敗: {e}")


# 主程式入口
def main_multi(goal_list, loop=False):
    global ouster_front

    threads_data = {
        "gps_port": "COM7",
        "gps_baudrate": 115200,
        "ahrs_port": "COM12",
        "ahrs_baudrate": 115200,
    }

    DataCollector.DataCollector(
        threads_data["gps_port"],
        threads_data["gps_baudrate"],
        threads_data["ahrs_port"],
        threads_data["ahrs_baudrate"],
        switch="open",
    )

    time_series = [None, None]
    yaw_series = [None, None]
    yaw_rate = 0
    pre_rudder = None
    new_rudder = 0
    rudder_count = 0

    if len(goal_list) > 1 and loop:
        goal_list.append(goal_list[0])
        print("Loop is open!")

    for i in range(len(goal_list)):
        goal_point = goal_list[i]
        print(f"point {goal_point} ({i + 1}/{len(goal_list)})")

        if (i + 1) == len(goal_list):
            final_point = True
            ouster_event.set()  # ✅ 啟動 Ouster 執行緒
        else:
            final_point = False
            ouster_event.clear()  # ✅ 停止 Ouster 執行緒

        while True:
            ship_data = {
                "Heading": gps_data["Heading"],
                "Lat": gps_data["Lat"],
                "Lon": gps_data["Lon"],
                "Roll": attitude_data["Roll"],
                "Pitch": attitude_data["Pitch"],
                "Yaw": attitude_data["Yaw"],
            }

            if all(value is not None for value in ship_data.values()):
                if keyboard.is_pressed("esc"):
                    print("Emergency Stop!")
                    DataTransmitter("stop")
                    time.sleep(2)
                    DataTransmitter("back")
                    time.sleep(5)
                    DataTransmitter("stop")
                    time.sleep(2)
                    os._exit(0)
                    break

                current_time = time.time()
                current_yaw = ship_data["Yaw"]

                time_series[0] = time_series[1]
                time_series[1] = current_time
                yaw_series[0] = yaw_series[1]
                yaw_series[1] = current_yaw

                if time_series[0] is not None and yaw_series[0] is not None:
                    try:
                        delta_yaw = np.diff(yaw_series)
                        delta_t = np.diff(time_series)
                        yaw_rate = delta_yaw[0] / delta_t[0] if delta_t[0] != 0 else 0
                    except Exception as e:
                        print(f"計算 yaw_rate 出錯: {e}")
                        yaw_rate = 0
                else:
                    yaw_rate = 0

                distance = Cal_Rudder_Engine.calculate_distance(
                    ship_data["Lat"], ship_data["Lon"], goal_point[0], goal_point[1]
                )

                rudder_calc, target_lat, target_lon, yaw_error = (
                    Cal_Rudder_Engine.RudderAngleCalculation(
                        ship_data,
                        goal_point,
                        yaw_rate if isinstance(yaw_rate, (float, int)) else 0,
                    )
                )

                new_rudder = rudder_calc
                if new_rudder == pre_rudder:
                    rudder_count += 0.34
                else:
                    rudder_count = 0

                rudder_calc += np.sign(new_rudder) * (
                    min(rudder_count, 20) if final_point else min(rudder_count, 24)
                )
                rudder_calc = int(max(min(rudder_calc, 25), -25))
                pre_rudder = new_rudder

                # 加入 Ouster 判斷：最後一個點時如果距離過近自動停止
                if final_point and ouster_front is not None and ouster_front <= 9.2:
                    print("Ouster 偵測到前方障礙物，停止靠泊")
                    ouster_event.clear()  # 停止 Ouster 執行緒
                    DataTransmitter("stop")
                    time.sleep(5)
                    DataTransmitter("left_back")
                    time.sleep(18)
                    DataTransmitter("stop")
                    time.sleep(2)
                    final_point = False
                    break
                print(f"Ouster:{ouster_front}")
                if (
                    rudder_calc == 0
                    and target_lat == 0
                    and target_lon == 0
                    and yaw_error == 0
                    and final_point
                ):
                    DataTransmitter("stop")
                    break
                elif (
                    rudder_calc == 0
                    and target_lat == 0
                    and target_lon == 0
                    and yaw_error == 0
                    and final_point == False
                ):
                    break
                # 寫入 CSV，包括 ouster_front 欄位
                with open(file_path, "a", newline="") as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(
                        [
                            datetime.now().strftime("%H:%M:%S"),
                            goal_point,
                            ship_data["Heading"],
                            ship_data["Lat"],
                            ship_data["Lon"],
                            ship_data["Roll"],
                            ship_data["Pitch"],
                            ship_data["Yaw"],
                            target_lat,
                            target_lon,
                            yaw_error,
                            rudder_calc,
                            distance,
                            ouster_front,
                        ]
                    )

                now_gps = ship_data["Lat"], ship_data["Lon"]
                target_point = [target_lat, target_lon]

                control_dict = {
                    "engine": Cal_Rudder_Engine.EngineCalculation(
                        now_gps, goal_point, final_point
                    ),
                    "rudder": rudder_calc,
                    "range": 0.05,
                }

                command_data = {
                    "Command": 666,
                    "Left_Speed": control_dict["engine"],
                    "Right_Speed": control_dict["engine"],
                    "Left_Rudder": control_dict["rudder"],
                    "Right_Rudder": control_dict["rudder"],
                    "Range": control_dict["range"],
                }

                DataTransmitter(command_data=command_data)

                print(
                    f"舵角:{control_dict['rudder']}  引擎:{control_dict['engine']}  升電壓:{control_dict['range']}"
                )
                print(f"Yaw: {ship_data['Yaw']}, Heading: {ship_data['Heading']}")
                print(
                    f"目前座標: {now_gps}\n目標點座標: {target_point}\n目標距離: {distance}"
                )

            # DataCollector.DataCollector(threads_data['gps_port'], threads_data['gps_baudrate'],threads_data['ahrs_port'], threads_data['ahrs_baudrate'],switch='open')


# 修正點
main_multi(
    [
        [22.864941252535015, 120.20133722574786],
        [22.865546389921825, 120.2007307623367],
    ],
    loop=False,
)  # 靠泊
