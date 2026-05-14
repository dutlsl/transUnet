import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 🚨 요청하신 1번 GPU 강제 할당
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# =====================================================================
# CSV 파일 경로 설정 (TransUNet 결과 파일로 지정)
BASE_CSV_PATH = "./Swirski_tables/transunet_swirski_scores.csv"
SAM_CSV_PATH = "./Swirski_tables/transunet_sam_swirski_scores.csv"

# 그래프 저장 폴더 (U-Mamba 결과와 겹치지 않게 분리)
OUTPUT_DIR = "./Swirski_plots_transunet"
# =====================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 데이터 로드
    if not os.path.exists(BASE_CSV_PATH) or not os.path.exists(SAM_CSV_PATH):
        print(f"🚨 CSV 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    df_base = pd.read_csv(BASE_CSV_PATH)
    df_sam = pd.read_csv(SAM_CSV_PATH)

    # 2. 요약 통계 행 필터링 (Frame_Idx가 소수점이나 문자가 아닌, '순수 정수'인 행만 유지)
    df_base = df_base[df_base['Frame_Idx'].astype(str).str.match(r'^\d+$')].copy()
    df_sam = df_sam[df_sam['Frame_Idx'].astype(str).str.match(r'^\d+$')].copy()

    # 3. 데이터 타입 변환
    for df in [df_base, df_sam]:
        df['Frame_Idx'] = df['Frame_Idx'].astype(int)
        df['IoU'] = df['IoU'].astype(float)
        df['Dice'] = df['Dice'].astype(float)

    # 4. Base와 SAM 데이터 병합 (동일한 Case와 Frame_Idx를 기준으로 매칭)
    df_merged = pd.merge(df_base, df_sam, on=['Case', 'Frame_Idx'], suffixes=('_Base', '_SAM'))

    cases = df_merged['Case'].unique()
    print(f"총 {len(cases)}개의 케이스를 발견했습니다. TransUNet 시각화를 시작합니다.")

    sns.set_theme(style="whitegrid")

    # 5. 케이스별 그래프 생성
    for case in cases:
        case_data = df_merged[df_merged['Case'] == case].sort_values('Frame_Idx')

        # --- [1] IoU 그래프 ---
        plt.figure(figsize=(12, 6))
        plt.plot(case_data['Frame_Idx'], case_data['IoU_Base'], label='TransUNet (Baseline)', color='tomato', marker='o', markersize=4, alpha=0.8)
        plt.plot(case_data['Frame_Idx'], case_data['IoU_SAM'], label='TransUNet + SAM', color='royalblue', marker='x', markersize=4, alpha=0.8)

        plt.title(f'[{case}] TransUNet Baseline vs SAM - IoU Score per Frame', fontsize=14, fontweight='bold')
        plt.xlabel('Frame Index', fontsize=12)
        plt.ylabel('IoU Score', fontsize=12)
        plt.ylim(0, 1.05) # 스코어 범위 고정
        plt.legend(fontsize=11)
        plt.tight_layout()

        iou_save_path = os.path.join(OUTPUT_DIR, f'{case}_IoU.png')
        plt.savefig(iou_save_path, dpi=300)
        plt.close()

        # --- [2] Dice 그래프 ---
        plt.figure(figsize=(12, 6))
        plt.plot(case_data['Frame_Idx'], case_data['Dice_Base'], label='TransUNet (Baseline)', color='tomato', marker='o', markersize=4, alpha=0.8)
        plt.plot(case_data['Frame_Idx'], case_data['Dice_SAM'], label='TransUNet + SAM', color='royalblue', marker='x', markersize=4, alpha=0.8)

        plt.title(f'[{case}] TransUNet Baseline vs SAM - Dice Score per Frame', fontsize=14, fontweight='bold')
        plt.xlabel('Frame Index', fontsize=12)
        plt.ylabel('Dice Score', fontsize=12)
        plt.ylim(0, 1.05)
        plt.legend(fontsize=11)
        plt.tight_layout()

        dice_save_path = os.path.join(OUTPUT_DIR, f'{case}_Dice.png')
        plt.savefig(dice_save_path, dpi=300)
        plt.close()

        print(f"  -> {case} 그래프 2종 저장 완료")

    print(f"\n모든 작업이 끝났습니다! 결과물은 '{OUTPUT_DIR}' 폴더를 확인해주세요.")

if __name__ == "__main__":
    main()