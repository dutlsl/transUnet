import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 🚨 1번 GPU 강제 할당
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# =====================================================================
# 4개의 CSV 파일 경로 설정
UMAMBA_BASE_CSV = "/mnt/ssd1/PycharmProjects/U-Mamba/LPW_tables/umamba_lpw_video_gt_scores.csv"
UMAMBA_SAM_CSV = "/mnt/ssd1/PycharmProjects/U-Mamba/LPW_tables/umamba_sam_lpw_video_gt_scores.csv"
TRANSUNET_BASE_CSV = "./LPW_tables/transunet_lpw_video_gt_scores.csv"
TRANSUNET_SAM_CSV = "./LPW_tables/transunet_sam_lpw_video_gt_scores.csv"

OUTPUT_DIR = "./LPW_plots_comparison"
FRAME_PLOTS_DIR = os.path.join(OUTPUT_DIR, "frame_plots")
# =====================================================================

def parse_lpw_csv(filepath, prefix):
    """파싱과 동시에 컬럼명에 접두사를 붙여 4개 데이터 병합 시 충돌을 막습니다."""
    frame_data, case_summary = [], []
    total_summary = {f'mIoU_{prefix}': 0.0}

    if not os.path.exists(filepath):
        return pd.DataFrame(), pd.DataFrame(), total_summary

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    mode = "frame"
    for line in lines:
        line = line.strip()
        if not line or line.replace(',', '') == '': continue

        if "=== CASE SUMMARY" in line: mode = "case_summary"; continue
        elif "=== TOTAL SUMMARY" in line: mode = "total_summary"; continue

        parts = line.split(',')

        if mode == "frame" and parts[0] != "Case" and len(parts) >= 5:
            frame_data.append({
                'Case': parts[0], 'Video_Name': parts[1], 'Frame_Idx': int(parts[2]),
                f'IoU_{prefix}': float(parts[3])
            })
        elif mode == "case_summary" and parts[0] != "Case" and len(parts) >= 3:
            case_summary.append({
                'Case': int(parts[0]), f'mIoU_{prefix}': float(parts[1])
            })
        elif mode == "total_summary" and parts[0] == "Total":
            total_summary[f'mIoU_{prefix}'] = float(parts[1])

    return pd.DataFrame(frame_data), pd.DataFrame(case_summary), total_summary

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FRAME_PLOTS_DIR, exist_ok=True)

    print("1. 4종 데이터 로드 및 파싱 중...")
    df_um_base_fr, df_um_base_sum, um_base_tot = parse_lpw_csv(UMAMBA_BASE_CSV, 'UM_Base')
    df_um_sam_fr, df_um_sam_sum, um_sam_tot = parse_lpw_csv(UMAMBA_SAM_CSV, 'UM_SAM')
    df_tu_base_fr, df_tu_base_sum, tu_base_tot = parse_lpw_csv(TRANSUNET_BASE_CSV, 'TU_Base')
    df_tu_sam_fr, df_tu_sam_sum, tu_sam_tot = parse_lpw_csv(TRANSUNET_SAM_CSV, 'TU_SAM')

    # 병합
    df_fr = pd.merge(df_um_base_fr, df_um_sam_fr, on=['Case', 'Video_Name', 'Frame_Idx'], how='outer')
    df_fr = pd.merge(df_fr, df_tu_base_fr, on=['Case', 'Video_Name', 'Frame_Idx'], how='outer')
    df_fr = pd.merge(df_fr, df_tu_sam_fr, on=['Case', 'Video_Name', 'Frame_Idx'], how='outer')

    sns.set_theme(style="whitegrid")
    video_groups = df_fr.groupby(['Case', 'Video_Name'])
    print(f"2. 총 {len(video_groups)}개 비디오의 4-Way 비교 그래프 생성을 시작합니다...")

    # [1] 프레임별 비교 그래프
    for (case_id, video_name), group in video_groups:
        group = group.sort_values('Frame_Idx')
        plt.figure(figsize=(14, 6))

        # U-Mamba (Blue)
        if 'IoU_UM_Base' in group: plt.plot(group['Frame_Idx'], group['IoU_UM_Base'], label='U-Mamba (Base)', color='royalblue', marker='o', markersize=3, alpha=0.9)
        if 'IoU_UM_SAM' in group: plt.plot(group['Frame_Idx'], group['IoU_UM_SAM'], label='U-Mamba + SAM', color='dodgerblue', linestyle='--', marker='x', markersize=3, alpha=0.9)

        # TransUNet (Red)
        if 'IoU_TU_Base' in group: plt.plot(group['Frame_Idx'], group['IoU_TU_Base'], label='TransUNet (Base)', color='crimson', marker='s', markersize=3, alpha=0.9)
        if 'IoU_TU_SAM' in group: plt.plot(group['Frame_Idx'], group['IoU_TU_SAM'], label='TransUNet + SAM', color='lightcoral', linestyle='--', marker='^', markersize=3, alpha=0.9)

        plt.title(f'[Case {case_id} - {video_name}] Model Comparison - IoU per Frame', fontsize=14, fontweight='bold')
        plt.xlabel('Frame Index', fontsize=12)
        plt.ylabel('IoU Score', fontsize=12)
        plt.ylim(0, 1.05)
        plt.legend(fontsize=11, loc='lower right')
        plt.tight_layout()

        filename = f"Case_{case_id}_{video_name.replace('.avi', '')}_IoU_Comparison.png"
        plt.savefig(os.path.join(FRAME_PLOTS_DIR, filename), dpi=300)
        plt.close()

    # [2] 케이스별 mIoU 요약 비교 그래프
    print("\n3. 케이스별 요약(Summary) 4-Way 그래프를 생성합니다...")
    df_sum = pd.merge(df_um_base_sum, df_um_sam_sum, on='Case', how='outer')
    df_sum = pd.merge(df_sum, df_tu_base_sum, on='Case', how='outer')
    df_sum = pd.merge(df_sum, df_tu_sam_sum, on='Case', how='outer').sort_values('Case')

    plt.figure(figsize=(16, 7))

    # 선 그래프
    if 'mIoU_UM_Base' in df_sum: plt.plot(df_sum['Case'].astype(str), df_sum['mIoU_UM_Base'], label='U-Mamba (Base)', color='royalblue', marker='o', linewidth=2, markersize=8)
    if 'mIoU_UM_SAM' in df_sum: plt.plot(df_sum['Case'].astype(str), df_sum['mIoU_UM_SAM'], label='U-Mamba + SAM', color='dodgerblue', linestyle='--', marker='x', linewidth=2, markersize=8)
    if 'mIoU_TU_Base' in df_sum: plt.plot(df_sum['Case'].astype(str), df_sum['mIoU_TU_Base'], label='TransUNet (Base)', color='crimson', marker='s', linewidth=2, markersize=8)
    if 'mIoU_TU_SAM' in df_sum: plt.plot(df_sum['Case'].astype(str), df_sum['mIoU_TU_SAM'], label='TransUNet + SAM', color='lightcoral', linestyle='--', marker='^', linewidth=2, markersize=8)

    # 4종류의 Total Score 점선 추가
    if 'mIoU_UM_Base' in um_base_tot: plt.axhline(y=um_base_tot['mIoU_UM_Base'], color='royalblue', linestyle=':', alpha=0.6, label=f"Total UM Base ({um_base_tot['mIoU_UM_Base']:.4f})")
    if 'mIoU_UM_SAM' in um_sam_tot: plt.axhline(y=um_sam_tot['mIoU_UM_SAM'], color='dodgerblue', linestyle='-.', alpha=0.6, label=f"Total UM SAM ({um_sam_tot['mIoU_UM_SAM']:.4f})")
    if 'mIoU_TU_Base' in tu_base_tot: plt.axhline(y=tu_base_tot['mIoU_TU_Base'], color='crimson', linestyle=':', alpha=0.6, label=f"Total TU Base ({tu_base_tot['mIoU_TU_Base']:.4f})")
    if 'mIoU_TU_SAM' in tu_sam_tot: plt.axhline(y=tu_sam_tot['mIoU_TU_SAM'], color='lightcoral', linestyle='-.', alpha=0.6, label=f"Total TU SAM ({tu_sam_tot['mIoU_TU_SAM']:.4f})")

    plt.title('LPW Dataset 4-Way Comparison: mIoU per Case', fontsize=18, fontweight='bold')
    plt.xlabel('Case ID', fontsize=12)
    plt.ylabel('mIoU Score', fontsize=12)
    plt.ylim(0, 1.05)

    # 레전드 위치를 바깥으로 빼서 선명하게
    plt.legend(fontsize=11, loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    summary_save_path = os.path.join(OUTPUT_DIR, 'LPW_All_Models_Case_Summary.png')
    plt.savefig(summary_save_path, dpi=300)
    plt.close()

    print(f"\n모든 작업이 끝났습니다! 결과물은 '{OUTPUT_DIR}' 폴더를 확인해주세요.")

if __name__ == "__main__":
    main()