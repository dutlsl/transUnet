import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 🚨 요청하신 1번 GPU 강제 할당
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# =====================================================================
# CSV 파일 경로 설정 (총 4개)
UMAMBA_BASE_CSV = "/mnt/ssd1/PycharmProjects/U-Mamba/Swirski_tables/umamba_swirski_scores.csv"
UMAMBA_SAM_CSV = "/mnt/ssd1/PycharmProjects/U-Mamba/Swirski_tables/sam1_swirski_scores.csv"
TRANSUNET_BASE_CSV = "./Swirski_tables/transunet_swirski_scores.csv"
TRANSUNET_SAM_CSV = "./Swirski_tables/transunet_sam_swirski_scores.csv"

# 그래프 저장 폴더
OUTPUT_DIR = "./Swirski_plots_comparison"
# =====================================================================

def load_and_clean_csv(file_path, prefix):
    """CSV를 불러오고, 불필요한 요약 통계 행을 제거한 뒤 컬럼명을 변경합니다."""
    if not os.path.exists(file_path):
        print(f"🚨 파일을 찾을 수 없습니다: {file_path}")
        return pd.DataFrame(columns=['Case', 'Frame_Idx'])

    df = pd.read_csv(file_path)
    # Frame_Idx가 순수 정수인 행만 남김 (요약 통계 방어)
    df = df[df['Frame_Idx'].astype(str).str.match(r'^\d+$')].copy()

    df['Frame_Idx'] = df['Frame_Idx'].astype(int)
    df['IoU'] = df['IoU'].astype(float)
    df['Dice'] = df['Dice'].astype(float)

    # 병합 시 충돌을 막기 위해 컬럼명 변경
    df = df.rename(columns={'IoU': f'IoU_{prefix}', 'Dice': f'Dice_{prefix}'})
    return df

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("1. 데이터 로드 및 정제 중...")
    df_um_base = load_and_clean_csv(UMAMBA_BASE_CSV, 'UM_Base')
    df_um_sam = load_and_clean_csv(UMAMBA_SAM_CSV, 'UM_SAM')
    df_tu_base = load_and_clean_csv(TRANSUNET_BASE_CSV, 'TU_Base')
    df_tu_sam = load_and_clean_csv(TRANSUNET_SAM_CSV, 'TU_SAM')

    # 2. 4개의 데이터프레임 병합 (Outer Join을 사용하여 누락된 프레임도 커버)
    print("2. 데이터 병합 중...")
    df_merged = pd.merge(df_um_base, df_um_sam, on=['Case', 'Frame_Idx'], how='outer')
    df_merged = pd.merge(df_merged, df_tu_base, on=['Case', 'Frame_Idx'], how='outer')
    df_merged = pd.merge(df_merged, df_tu_sam, on=['Case', 'Frame_Idx'], how='outer')

    cases = df_merged['Case'].dropna().unique()
    print(f"총 {len(cases)}개의 케이스를 발견했습니다. 그래프 생성을 시작합니다.")

    sns.set_theme(style="whitegrid")

    # 3. 케이스별 그래프 생성
    for case in cases:
        case_data = df_merged[df_merged['Case'] == case].sort_values('Frame_Idx')

        # --- [1] IoU 그래프 ---
        plt.figure(figsize=(14, 7))

        # U-Mamba (Blue 계열)
        if 'IoU_UM_Base' in case_data:
            plt.plot(case_data['Frame_Idx'], case_data['IoU_UM_Base'], label='U-Mamba (Base)', color='royalblue', linestyle='-', marker='o', markersize=5, alpha=0.9)
        if 'IoU_UM_SAM' in case_data:
            plt.plot(case_data['Frame_Idx'], case_data['IoU_UM_SAM'], label='U-Mamba + SAM', color='dodgerblue', linestyle='--', marker='x', markersize=5, alpha=0.9)

        # TransUNet (Red 계열)
        if 'IoU_TU_Base' in case_data:
            plt.plot(case_data['Frame_Idx'], case_data['IoU_TU_Base'], label='TransUNet (Base)', color='crimson', linestyle='-', marker='s', markersize=5, alpha=0.9)
        if 'IoU_TU_SAM' in case_data:
            plt.plot(case_data['Frame_Idx'], case_data['IoU_TU_SAM'], label='TransUNet + SAM', color='lightcoral', linestyle='--', marker='^', markersize=5, alpha=0.9)

        plt.title(f'[{case}] Model Comparison - IoU Score per Frame', fontsize=16, fontweight='bold')
        plt.xlabel('Frame Index', fontsize=12)
        plt.ylabel('IoU Score', fontsize=12)
        plt.ylim(0, 1.05)
        plt.legend(fontsize=11, loc='lower right')
        plt.tight_layout()

        iou_save_path = os.path.join(OUTPUT_DIR, f'{case}_IoU_Comparison.png')
        plt.savefig(iou_save_path, dpi=300)
        plt.close()

        # --- [2] Dice 그래프 ---
        plt.figure(figsize=(14, 7))

        if 'Dice_UM_Base' in case_data:
            plt.plot(case_data['Frame_Idx'], case_data['Dice_UM_Base'], label='U-Mamba (Base)', color='royalblue', linestyle='-', marker='o', markersize=5, alpha=0.9)
        if 'Dice_UM_SAM' in case_data:
            plt.plot(case_data['Frame_Idx'], case_data['Dice_UM_SAM'], label='U-Mamba + SAM', color='dodgerblue', linestyle='--', marker='x', markersize=5, alpha=0.9)

        if 'Dice_TU_Base' in case_data:
            plt.plot(case_data['Frame_Idx'], case_data['Dice_TU_Base'], label='TransUNet (Base)', color='crimson', linestyle='-', marker='s', markersize=5, alpha=0.9)
        if 'Dice_TU_SAM' in case_data:
            plt.plot(case_data['Frame_Idx'], case_data['Dice_TU_SAM'], label='TransUNet + SAM', color='lightcoral', linestyle='--', marker='^', markersize=5, alpha=0.9)

        plt.title(f'[{case}] Model Comparison - Dice Score per Frame', fontsize=16, fontweight='bold')
        plt.xlabel('Frame Index', fontsize=12)
        plt.ylabel('Dice Score', fontsize=12)
        plt.ylim(0, 1.05)
        plt.legend(fontsize=11, loc='lower right')
        plt.tight_layout()

        dice_save_path = os.path.join(OUTPUT_DIR, f'{case}_Dice_Comparison.png')
        plt.savefig(dice_save_path, dpi=300)
        plt.close()

        print(f"  -> {case} 비교 그래프 2종 저장 완료")

    print(f"\n모든 작업이 끝났습니다! 결과물은 '{OUTPUT_DIR}' 폴더를 확인해주세요.")

if __name__ == "__main__":
    main()