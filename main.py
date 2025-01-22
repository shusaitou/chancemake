import yaml
import pandas as pd
import numpy as np
import math

def load_yaml_to_dataframe(filepath: str) -> pd.DataFrame:
    """
    YAMLファイルを読み込み、Pandas DataFrameに変換して返す。
    実際のデータ構造に合わせて修正が必要。
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    # ここでは仮に data['events'] というリストが存在し、
    # 1つ1つが { 'id': ..., 'type': ..., 'x': ..., 'y': ..., ... } の形を想定。
    events = data.get('events', [])
    df = pd.DataFrame(events)
    
    # 例：もし 'geometry' の中に x, y が入っている等なら展開する必要がある
    # df['x'] = df['geometry'].apply(lambda g: g['x'])
    # df['y'] = df['geometry'].apply(lambda g: g['y'])
    
    return df

def compute_shot_index(data: pd.DataFrame,
                       pitch_length=105,
                       pitch_width=68,
                       num_zones_x=20,
                       num_zones_y=10) -> np.ndarray:
    """
    ショットデータ全体を用いて「ゾーンごとのゴール確率」を計算し、shot_index(2次元配列)を返す。
    data: 全イベントのDataFrame。 'event_type' == 'shot' がシュート。
          'start_x', 'start_y', 'outcome' (Goal or Missなど) が含まれている想定。
    """
    # ゾーン幅
    zone_width = pitch_length / num_zones_x
    zone_height = pitch_width / num_zones_y
    
    # シュートイベント抽出
    shot_data = data[data['event_type'] == 'shot']
    
    goal_counts = np.zeros((num_zones_y, num_zones_x))
    shot_counts = np.zeros((num_zones_y, num_zones_x))
    
    for _, row in shot_data.iterrows():
        x = row['start_x']
        y = row['start_y']
        zone_x = int(x / zone_width)
        zone_y = int(y / zone_height)
        
        # ゾーン外に行かないようにクリップ
        if zone_x >= num_zones_x:
            zone_x = num_zones_x - 1
        if zone_y >= num_zones_y:
            zone_y = num_zones_y - 1
        
        # outcomeがGoalならゴール数を加算
        if 'outcome' in row and row['outcome'] == 'Goal':
            goal_counts[zone_y][zone_x] += 1
        
        shot_counts[zone_y][zone_x] += 1
    
    # ゾーンごとのゴール確率
    # shot_counts!=0 の部分だけ計算し、そうでない部分は 0 とする
    shot_index = np.divide(
        goal_counts,
        shot_counts,
        out=np.zeros_like(goal_counts),
        where=(shot_counts!=0)
    )
    return shot_index

def calculate_features_for_possession(possession_data: pd.DataFrame,
                                      shot_index: np.ndarray,
                                      pitch_length=105,
                                      pitch_width=68,
                                      num_zones_x=20,
                                      num_zones_y=10) -> dict:
    """
    ユーザ様が示された18の指標を possession_data (当該ポゼッションのイベント配列) に基づいて計算し、
    dict にまとめて返す。
    
    possession_data: 該当ポゼッションの DataFrame
        必須列例:
        - 'x', 'y' : ボール移動の全座標
        - 'start_x', 'start_y', 'end_x', 'end_y' : パス等の開始/終了座標
        - 'event_type' : 'pass', 'shot' などのイベント種別
        - 'timestamp' : イベント時刻 または（開始時刻など）
        - 'player_id' : イベントの実行者
        - 'event_duration' : イベントの継続時間 (保持時間など)
        などなど
        
    shot_index: ショットインデックス (compute_shot_index関数で取得した2次元配列)
    """
    # --- 前提値 ---
    # ゴール座標
    goal_x = 105
    goal_y = 34
    
    # ゾーンの幅と高さ
    zone_width = pitch_length / num_zones_x
    zone_height = pitch_width / num_zones_y
    
    # === データ準備 ===
    # ポゼッション内の全座標列を配列化 (x,y)
    path_points = possession_data[['x','y']].dropna().values  # 欠損を除いてリスト化
    if len(path_points) < 2:
        # path_pointsが足りない場合はゼロ埋めなど適宜
        return {
            "overall_path_length": 0, 
            "ratio_of_passing_length": 0,
            "shot_distance": 0,
            "start_distance": 0,
            "average_distance": 0,
            "x_range": 0,
            "y_range": 0,
            "moving_directness": 0,
            "possession_time": 0,
            "overall_moving_speed": 0,
            "direct_speed": 0,
            "passing_ratio": 0,
            "acceleration_index": 0,
            "attack_intensity": 0,
            "number_of_players": 0,
            "centralization_of_passing_actions": 0,
            "centralization_of_possession_time": 0
        }
    
    # パスイベント抽出
    pass_events = possession_data[possession_data['event_type'] == 'pass']
    # シュートイベント抽出（該当ポゼッション内にシュートが1つもない場合もありうる）
    shot_events = possession_data[possession_data['event_type'] == 'shot']
    
    # === 1. 全体のパス経路長 ===
    overall_path_length = 0
    for i in range(1, len(path_points)):
        dx = path_points[i][0] - path_points[i-1][0]
        dy = path_points[i][1] - path_points[i-1][1]
        dist = np.sqrt(dx**2 + dy**2)
        overall_path_length += dist
    
    # === 2. パス距離の比率 ===
    pass_length = 0
    for _, row in pass_events.iterrows():
        dx = row['end_x'] - row['start_x']
        dy = row['end_y'] - row['start_y']
        dist = np.sqrt(dx**2 + dy**2)
        pass_length += dist
    
    ratio_of_passing_length = 0
    if overall_path_length != 0:
        ratio_of_passing_length = pass_length / overall_path_length
    
    # === 3. シュート距離 ===
    # ここでは最初のシュート(shot_event)のみを取得する例
    # 該当ポゼッションにシュートが無い場合を考慮
    shot_distance = 0
    if len(shot_events) > 0:
        shot_event = shot_events.iloc[0]
        dx = shot_event['start_x'] - goal_x
        dy = shot_event['start_y'] - goal_y
        shot_distance = np.sqrt(dx**2 + dy**2)
    
    # === 4. スタート距離 (ポゼッション開始点からゴールまで) ===
    start_x = path_points[0][0]
    start_y = path_points[0][1]
    dx = start_x - goal_x
    dy = start_y - goal_y
    start_distance = np.sqrt(dx**2 + dy**2)
    
    # === 5. 平均距離 ===
    distances = []
    for (x, y) in path_points:
        dx = x - goal_x
        dy = y - goal_y
        dist = np.sqrt(dx**2 + dy**2)
        distances.append(dist)
    average_distance = np.mean(distances) if len(distances) > 0 else 0
    
    # === 6. Xレンジ ===
    x_values = path_points[:, 0]
    x_range = x_values.max() - x_values.min()
    
    # === 7. Yレンジ ===
    y_values = path_points[:, 1]
    y_range = y_values.max() - y_values.min()
    
    # === 8. 移動の直進性 (移動開始点と終了点の直線距離 / 全体のパス経路長) ===
    dx = path_points[-1][0] - path_points[0][0]
    dy = path_points[-1][1] - path_points[0][1]
    straight_line_distance = np.sqrt(dx**2 + dy**2)
    
    moving_directness = 0
    if overall_path_length != 0:
        moving_directness = straight_line_distance / overall_path_length
    
    # === 9. ポゼッションタイム ===
    start_time = possession_data['timestamp'].iloc[0]
    end_time = possession_data['timestamp'].iloc[-1]
    possession_time = end_time - start_time
    
    # === 10. 全体の移動速度 ===
    # overall_moving_speed = 全体の移動距離 / ポゼッション時間
    overall_moving_speed = 0
    if possession_time != 0:
        overall_moving_speed = overall_path_length / possession_time
    
    # === 11. 直線速度 ===
    direct_speed = 0
    if possession_time != 0:
        direct_speed = straight_line_distance / possession_time
    
    # === 12. パス比率 (単位時間あたりのパス数)
    number_of_passes = len(pass_events)
    passing_ratio = 0
    if possession_time != 0:
        passing_ratio = number_of_passes / possession_time
    
    # === 13. ショットインデックスを用いた 加速度インデックス ===
    #   (shooting_area_shot_index / (possession_time^2))
    #   該当ポゼッションにシュートがない場合は0とする
    
    acceleration_index = 0
    if len(shot_events) > 0 and possession_time != 0:
        shot_event = shot_events.iloc[0]  # 最初のシュートのみ
        shot_x = shot_event['start_x']
        shot_y = shot_event['start_y']
        zone_x = int(shot_x / zone_width)
        zone_y = int(shot_y / zone_height)
        if zone_x >= num_zones_x:
            zone_x = num_zones_x - 1
        if zone_y >= num_zones_y:
            zone_y = num_zones_y - 1
        
        shooting_area_shot_index = shot_index[zone_y][zone_x]
        acceleration_index = shooting_area_shot_index / (possession_time ** 2)
    
    # === 14. 攻撃強度: path_points の各点での shot_index の合計 / possession_time ===
    total_shot_index = 0
    for (px, py) in path_points:
        zx = int(px / zone_width)
        zy = int(py / zone_height)
        if zx >= num_zones_x:
            zx = num_zones_x - 1
        if zy >= num_zones_y:
            zy = num_zones_y - 1
        total_shot_index += shot_index[zy][zx]
    
    attack_intensity = 0
    if possession_time != 0:
        attack_intensity = total_shot_index / possession_time
    
    # === 15. プレイヤー数 (関与した全プレイヤーID数) ===
    players_involved = possession_data['player_id'].unique()
    number_of_players = len(players_involved)
    
    # === 16. パスアクションの集中度 (centralization_of_passing_actions) ===
    pass_events = possession_data[possession_data['event_type'] == 'pass']
    pass_counts_series = pass_events['player_id'].value_counts()
    
    if len(pass_counts_series) > 0:
        N_p_star = pass_counts_series.max()
        # 分母を 0 で割らないように注意
        n = len(pass_counts_series)
        
        numerator = sum(N_p_star - pass_counts_series)
        denominator = N_p_star * (n - 1)
        
        if denominator != 0:
            centralization_of_passing_actions = numerator / denominator
        else:
            centralization_of_passing_actions = 0
    else:
        centralization_of_passing_actions = 0
    
    # === 17. ボール保持時間の集中度 (centralization_of_possession_time) ===
    # 各プレイヤーのボール保持時間 = event_duration の合計
    possession_time_by_player = possession_data.groupby('player_id')['event_duration'].sum()
    
    if len(possession_time_by_player) > 0:
        T_p_star = possession_time_by_player.max()
        n = len(possession_time_by_player)
        
        numerator = sum(T_p_star - possession_time_by_player)
        denominator = T_p_star * (n - 1)
        
        if denominator != 0:
            centralization_of_possession_time = numerator / denominator
        else:
            centralization_of_possession_time = 0
    else:
        centralization_of_possession_time = 0
    
    # === 結果まとめ ===
    features = {
        "overall_path_length": overall_path_length,
        "ratio_of_passing_length": ratio_of_passing_length,
        "shot_distance": shot_distance,
        "start_distance": start_distance,
        "average_distance": average_distance,
        "x_range": x_range,
        "y_range": y_range,
        "moving_directness": moving_directness,
        "possession_time": possession_time,
        "overall_moving_speed": overall_moving_speed,
        "direct_speed": direct_speed,
        "passing_ratio": passing_ratio,
        "acceleration_index": acceleration_index,
        "attack_intensity": attack_intensity,
        "number_of_players": number_of_players,
        "centralization_of_passing_actions": centralization_of_passing_actions,
        "centralization_of_possession_time": centralization_of_possession_time
    }
    
    return features

def main():
    # 1. YAMLファイルを読み込み、DataFrameに変換
    all_data = load_yaml_to_dataframe('current.yml')
    
    # ---------------------------------------------------------------------
    #  下記はデータ例。実際には、列名やデータ構造をユーザ様の状況に合わせて整備してください。
    #
    #  想定される列例：
    #   - 'event_type' : (pass, shot, carry, etc.)
    #   - 'x', 'y' : イベントごとの位置(ボール位置)
    #   - 'start_x','start_y','end_x','end_y' : pass等の開始終了座標
    #   - 'timestamp' : イベント発生時刻（秒またはミリ秒）
    #   - 'player_id' : イベントを実行した選手ID
    #   - 'event_duration' : イベントの継続時間
    #   - 'outcome' : シュートの結果（Goal / Miss / ...）
    #   - 'possession_id' : ポゼッションを一意に示すID
    #
    #  注意: 上記列がそろわないと、各種計算が正常に動作しません。
    # ---------------------------------------------------------------------
    
    # 例）必要に応じて欠損値の補完や変換を行う
    all_data = all_data.fillna({
        'start_x': 0, 'start_y': 0, 'end_x': 0, 'end_y': 0,
        'x': 0, 'y': 0, 'event_duration': 0
    })
    
    # 2. ショットインデックスを算出する(試合全体から計算)
    shot_idx_2d = compute_shot_index(all_data)
    
    # 3. 分析したいポゼッション(あるいは複数のpossession_id)を取得して、18特徴を算出
    #    ここでは例として、possession_id=1 のポゼッションだけ取り出すと仮定
    #    実際にはループで全possession_idを回すなどの使い方もできます
    target_possession_id = 1
    possession_data = all_data[all_data['possession_id'] == target_possession_id].copy()
    
    # タイムスタンプで並び替えるなどしておく
    possession_data.sort_values(by='timestamp', inplace=True)
    
    # 18指標の計算
    features_dict = calculate_features_for_possession(
        possession_data=possession_data, 
        shot_index=shot_idx_2d
    )
    
    # 4. 結果の出力
    print("=== Features for possession_id =", target_possession_id, "===")
    for k, v in features_dict.items():
        print(f"{k}: {v}")

    print(all_data.columns)



if __name__ == '__main__':
    main()