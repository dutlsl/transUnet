import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 🚨 1번 GPU 강제 할당
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# =====================================================================
# CSV 파일 경로 설정 (TransUNet 파일로 지정)
BASE_CSV_PATH = "./LPW_tables/transunet_lpw_video_gt_scores.csv"
SAM_CSV_PATH = "./LPW_tables/transunet_sam_lpw_video_gt_scores.csv"

# 그래프 저장 폴더
OUTPUT_DIR = "./LPW_plots_transunet"
FRAME_PLOTS_DIR = os.path.join(OUTPUT_DIR, "frame_plots")
# =====================================================================

def parse_lpw_csv(filepath):
    """CSV 파일을 읽어 프레임 데이터와 요약 데이터를 분리하여 파싱합니다."""
    frame_data, case_summary = [], []
    total_summary = {'mIoU': 0.0, 'mDice': 0.0}

    if not os.path.exists(filepath):
        print(f"🚨 파일을 찾을 수 없습니다: {filepath}")
        return pd.DataFrame(), pd.DataFrame(), total_summary

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    mode = "frame"
    for line in lines:
        line = line.strip()
        # 빈 줄이거나 쉼표만 있는 줄은 무시
        if not line or line.replace(',', '') == '':
            continue

        if "=== CASE SUMMARY" in line:
            mode = "case_summary"
            continue
        elif "=== TOTAL SUMMARY" in line:
            mode = "total_summary"
            continue

        parts = line.split(',')

        if mode == "frame":
            if parts[0] == "Case": continue
            if len(parts) >= 5:
                frame_data.append({
                    'Case': parts[0],
                    'Video_Name': parts[1],
                    'Frame_Idx': int(parts[2]),
                    'IoU': float(parts[3]),
                    'Dice': float(parts[4])
                })
        elif mode == "case_summary":
            if parts[0] == "Case": continue
            if len(parts) >= 3:
                case_summary.append({
                    'Case': int(parts[0]), # 정렬을 위해 정수로 변환
                    'mIoU': float(parts[1]),
                    'mDice': float(parts[2])
                })
        elif mode == "total_summary":
            if parts[0] == "Total":
                total_summary = {'mIoU': float(parts[1]), 'mDice': float(parts[2])}

    return pd.DataFrame(frame_data), pd.DataFrame(case_summary), total_summary

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FRAME_PLOTS_DIR, exist_ok=True)

    print("1. 데이터 로드 및 파싱 중...")
    df_base_frame, df_base_summary, base_total = parse_lpw_csv(BASE_CSV_PATH)
    df_sam_frame, df_sam_summary, sam_total = parse_lpw_csv(SAM_CSV_PATH)

    if df_base_frame.empty or df_sam_frame.empty:
        print("🚨 유효한 데이터가 없습니다. CSV 파일을 확인해주세요.")
        return

    sns.set_theme(style="whitegrid")

    # ==========================================================
    # [1] 프레임별 IoU 그래프 생성 (각 비디오별)
    # ==========================================================
    df_merged_frame = pd.merge(
        df_base_frame, df_sam_frame,
        on=['Case', 'Video_Name', 'Frame_Idx'],
        suffixes=('_Base', '_SAM')
    )

    # (Case, Video_Name) 쌍으로 그룹화
    video_groups = df_merged_frame.groupby(['Case', 'Video_Name'])
    print(f"2. 총 {len(video_groups)}개 비디오의 프레임별 그래프 생성을 시작합니다...")

    for (case_id, video_name), group in video_groups:
        group = group.sort_values('Frame_Idx')

        plt.figure(figsize=(12, 5))
        plt.plot(group['Frame_Idx'], group['IoU_Base'], label='TransUNet (Baseline)', color='tomato', marker='o', markersize=3, alpha=0.8)
        plt.plot(group['Frame_Idx'], group['IoU_SAM'], label='TransUNet + SAM', color='royalblue', marker='x', markersize=3, alpha=0.8)

        plt.title(f'[Case {case_id} - {video_name}] TransUNet Baseline vs SAM - IoU per Frame', fontsize=14, fontweight='bold')
        plt.xlabel('Frame Index', fontsize=12)
        plt.ylabel('IoU Score', fontsize=12)
        plt.ylim(0, 1.05)
        plt.legend(fontsize=11)
        plt.tight_layout()

        filename = f"Case_{case_id}_{video_name.replace('.avi', '')}_IoU.png"
        plt.savefig(os.path.join(FRAME_PLOTS_DIR, filename), dpi=300)
        plt.close()

    # ==========================================================
    # [2] 케이스(폴더)별 mIoU 요약 그래프 생성 (이전 누락분 복구)
    # ==========================================================
    print("\n3. 케이스별 요약(Summary) 그래프를 생성합니다...")

    df_merged_summary = pd.merge(
        df_base_summary, df_sam_summary,
        on='Case',
        suffixes=('_Base', '_SAM')
    ).sort_values('Case') # Case 번호순으로 깔끔하게 정렬

    plt.figure(figsize=(14, 6))

    # 선 그래프로 가독성 높게 표기
    plt.plot(df_merged_summary['Case'].astype(str), df_merged_summary['mIoU_Base'],
             label='TransUNet (mIoU)', color='tomato', marker='o', linewidth=2, markersize=8)
    plt.plot(df_merged_summary['Case'].astype(str), df_merged_summary['mIoU_SAM'],
             label='TransUNet + SAM (mIoU)', color='royalblue', marker='s', linewidth=2, markersize=8)

    # 💡 [핵심] Total Score 점선(가로선) 추가
    plt.axhline(y=base_total['mIoU'], color='tomato', linestyle='--', alpha=0.6,
                label=f"Total Baseline ({base_total['mIoU']:.4f})")
    plt.axhline(y=sam_total['mIoU'], color='royalblue', linestyle='--', alpha=0.6,
                label=f"Total SAM ({sam_total['mIoU']:.4f})")

    plt.title('LPW Dataset (TransUNet): mIoU Comparison per Case', fontsize=16, fontweight='bold')
    plt.xlabel('Case ID', fontsize=12)
    plt.ylabel('mIoU Score', fontsize=12)
    plt.ylim(0, 1.05)

    # 점선 라벨이 잘 보이도록 레전드 위치 조정
    plt.legend(fontsize=11, loc='lower right', bbox_to_anchor=(1, 0.05))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    summary_save_path = os.path.join(OUTPUT_DIR, 'LPW_TransUNet_Case_Summary_mIoU.png')
    plt.savefig(summary_save_path, dpi=300)
    plt.close()

    print(f"\n모든 작업이 끝났습니다! 결과물은 '{OUTPUT_DIR}' 폴더를 확인해주세요.")

if __name__ == "__main__":
    main()