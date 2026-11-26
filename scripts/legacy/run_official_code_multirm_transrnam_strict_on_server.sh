#!/usr/bin/env bash
set -euo pipefail

# Official-code baseline pipeline for MultiRM and TransRNAm on this project's
# RMBase MultiRM-aligned labels and strict CD-HIT split.
#
# Usage on server, from project root:
#
#   env -u LD_PRELOAD bash scripts/run_official_code_multirm_transrnam_strict_on_server.sh multirm12 101 0
#
# Arguments:
#   1) preset: multirm12 or extended13
#   2) window length: 51 or 101
#   3) physical CUDA id
#
# If hg38.fa is not present, set DOWNLOAD_HG38=1 to download it from UCSC:
#
#   DOWNLOAD_HG38=1 env -u LD_PRELOAD bash scripts/run_official_code_multirm_transrnam_strict_on_server.sh multirm12 101 0

unset LD_PRELOAD || true

PRESET="${1:-multirm12}"
WINDOW_LENGTH="${2:-101}"
CUDA_ID="${3:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "${PRESET}" == "multirm12" ]]; then
  CLASS_NAMES="Am,Cm,Gm,Um,m1A,m5C,m5U,m6A,m6Am,m7G,Y,AtoI"
elif [[ "${PRESET}" == "extended13" ]]; then
  CLASS_NAMES="Am,Cm,Gm,Um,D,Y,m1A,m5C,m5U,m6A,m6Am,m7G,AtoI"
else
  echo "Unknown preset: ${PRESET}. Use multirm12 or extended13." >&2
  exit 2
fi
NUM_CLASSES="$("${PYTHON_BIN}" - <<PY
print(len("${CLASS_NAMES}".split(",")))
PY
)"

if [[ "${WINDOW_LENGTH}" != "51" && "${WINDOW_LENGTH}" != "101" ]]; then
  echo "WINDOW_LENGTH should be 51 or 101 for official-code baselines." >&2
  exit 2
fi

THREADS="${THREADS:-32}"
OFFICIAL_EPOCHS="${OFFICIAL_EPOCHS:-50}"
MULTIRM_BATCH_SIZE="${MULTIRM_BATCH_SIZE:-128}"
TRANSRNAM_BATCH_SIZE="${TRANSRNAM_BATCH_SIZE:-128}"
DOWNLOAD_HG38="${DOWNLOAD_HG38:-0}"

MULTIRM_DIR="${MULTIRM_DIR:-external/MultiRM-master}"
TRANSRNAM_SOURCE_DIR="${TRANSRNAM_SOURCE_DIR:-rmbase11_full_work/baseline_repos/TransRNAm-main}"
WORK_ROOT="${WORK_ROOT:-rmbase12_work/official_code_reproduction/${PRESET}_${WINDOW_LENGTH}nt}"
REFERENCE_DIR="${REFERENCE_DIR:-rmbase12_work/reference}"
GENOME_FASTA="${GENOME_FASTA:-${REFERENCE_DIR}/hg38.fa}"

BASE_DATASET="rmbase12_work/datasets/rmbase_${PRESET}_hg38_41nt_all_sites.tsv"
WINDOW_DATASET="rmbase12_work/datasets/rmbase_${PRESET}_hg38_51_101nt_all_sites.tsv"
STRICT_DATASET="rmbase12_work/datasets/rmbase_${PRESET}_hg38_${WINDOW_LENGTH}nt_cdhit80_strict_splits.tsv"
STRICT_SUMMARY="rmbase12_work/datasets/rmbase_${PRESET}_${WINDOW_LENGTH}nt_cdhit80_strict_splits_summary.json"
CDHIT_FASTA="rmbase12_work/cluster/cdhit80_${WINDOW_LENGTH}nt/rmbase_${PRESET}_${WINDOW_LENGTH}nt.fa"
CDHIT_PREFIX="rmbase12_work/cluster/cdhit80_${WINDOW_LENGTH}nt/rmbase_${PRESET}_${WINDOW_LENGTH}nt_cdhit80"
OFFICIAL_INPUT_DIR="rmbase12_work/official_code_inputs/${PRESET}_${WINDOW_LENGTH}nt_cdhit80"
OFFICIAL_H5="${OFFICIAL_INPUT_DIR}/data_12RM.h5"
TRANSRNAM_JSON_DIR="${OFFICIAL_INPUT_DIR}/transrnam_json"

mkdir -p "${WORK_ROOT}" "${REFERENCE_DIR}" "${OFFICIAL_INPUT_DIR}"
PROJECT_ROOT="$(pwd -P)"
WORK_ROOT_ABS="$(cd "${WORK_ROOT}" && pwd -P)"

echo "[0/8] Check tools"
command -v cd-hit-est >/dev/null || { echo "cd-hit-est not found in PATH" >&2; exit 1; }
"${PYTHON_BIN}" - <<'PY'
import importlib
missing = []
for name in ["numpy", "pandas", "tables", "h5py"]:
    try:
        importlib.import_module(name)
    except ModuleNotFoundError:
        missing.append(name)
if missing:
    raise SystemExit(
        "Missing Python packages: "
        + ", ".join(missing)
        + "\nInstall in the active environment, for example:\n"
        + "  conda install -y -c conda-forge "
        + " ".join(missing)
        + "\nOr:\n"
        + "  python -m pip install "
        + " ".join(missing)
    )
print("python dependencies ok")
PY

if [[ ! -s "${GENOME_FASTA}" ]]; then
  if [[ "${DOWNLOAD_HG38}" != "1" ]]; then
    cat >&2 <<EOF
Missing hg38 FASTA: ${GENOME_FASTA}
Set DOWNLOAD_HG38=1 to download UCSC hg38.fa.gz, or set GENOME_FASTA=/path/to/hg38.fa.
EOF
    exit 1
  fi
  echo "[1/8] Download hg38 FASTA from UCSC"
  mkdir -p "${REFERENCE_DIR}"
  if command -v wget >/dev/null; then
    wget -c -O "${REFERENCE_DIR}/hg38.fa.gz" "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
  else
    curl -L -C - -o "${REFERENCE_DIR}/hg38.fa.gz" "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
  fi
  gzip -dc "${REFERENCE_DIR}/hg38.fa.gz" > "${GENOME_FASTA}"
else
  echo "[1/8] Reuse hg38 FASTA: ${GENOME_FASTA}"
fi

echo "[2/8] Build ${PRESET} 41nt RMBase dataset if needed"
if [[ ! -s "${BASE_DATASET}" ]]; then
  "${PYTHON_BIN}" -B scripts/build_rmbase_multirm_aligned_dataset.py \
    --preset "${PRESET}" \
    --source-dirs rmbase11_work/extracted rmbase12_work/extracted \
    --out-dir rmbase12_work \
    --download-missing \
    --insecure-download
else
  echo "Reuse ${BASE_DATASET}"
fi

echo "[3/8] Add 51nt and 101nt windows from hg38 FASTA"
if [[ ! -s "${WINDOW_DATASET}" ]]; then
  "${PYTHON_BIN}" -B scripts/extract_rmbase_windows_from_fasta.py \
    --dataset "${BASE_DATASET}" \
    --fasta "${GENOME_FASTA}" \
    --out "${WINDOW_DATASET}" \
    --lengths 51,101 \
    --validate-existing-41 \
    --drop-invalid
else
  echo "Reuse ${WINDOW_DATASET}"
fi

echo "[4/8] Create ${WINDOW_LENGTH}nt CD-HIT80 strict split"
if [[ ! -s "${STRICT_DATASET}" ]]; then
  "${PYTHON_BIN}" -B scripts/create_rmbase11_cdhit_cluster_split.py \
    --dataset "${WINDOW_DATASET}" \
    --out "${STRICT_DATASET}" \
    --summary "${STRICT_SUMMARY}" \
    --fasta "${CDHIT_FASTA}" \
    --cdhit-output-prefix "${CDHIT_PREFIX}" \
    --sequence-column "sequence_${WINDOW_LENGTH}nt" \
    --cdhit-est "$(command -v cd-hit-est)" \
    --identity 0.8 \
    --word-size 5 \
    --coverage-shorter 0.8 \
    --coverage-longer 0.8 \
    --threads "${THREADS}" \
    --memory-mb 0 \
    --deduplicate-identical-sequences \
    --class-names "${CLASS_NAMES}"
else
  echo "Reuse ${STRICT_DATASET}"
fi

echo "[5/8] Convert strict split to official MultiRM HDF and TransRNAm JSON"
if [[ ! -s "${OFFICIAL_H5}" ]]; then
  "${PYTHON_BIN}" -B scripts/convert_rmbase_to_official_multirm_hdf.py \
    --dataset "${STRICT_DATASET}" \
    --split split_cdhit80_cluster \
    --sequence-column "sequence_${WINDOW_LENGTH}nt" \
    --class-names "${CLASS_NAMES}" \
    --embedding-path "${MULTIRM_DIR}/Embeddings/embeddings_12RM.pkl" \
    --out-h5 "${OFFICIAL_H5}" \
    --transrnam-json-dir "${TRANSRNAM_JSON_DIR}"
else
  echo "Reuse ${OFFICIAL_H5}"
fi
OFFICIAL_H5_ABS="$(cd "$(dirname "${OFFICIAL_H5}")" && pwd -P)/$(basename "${OFFICIAL_H5}")"

echo "[5.5/8] Patch optional plotting imports in official code"
"${PYTHON_BIN}" - <<PY
from pathlib import Path

def patch_optional_plotting(script_root: Path) -> None:
    train_utils = script_root / "train_utils.py"
    if train_utils.exists():
        text = train_utils.read_text(encoding="utf-8", errors="replace")
        old = """import matplotlib.pyplot as plt
from itertools import cycle
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
"""
        new = """from itertools import cycle
try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import matplotlib as mpl
    from matplotlib.font_manager import FontProperties
except Exception:
    plt = None
    ListedColormap = None
    mpl = None
    FontProperties = None
"""
        if old in text:
            text = text.replace(old, new)
        old_rmdata = """        if self.mode == 'train':
            self.train_x = pd.read_hdf(self.data_path,'train_in')
            self.train_y = pd.read_hdf(self.data_path,'train_out').to_numpy()
            self.valid_x = pd.read_hdf(self.data_path,'valid_in')
            self.valid_y = pd.read_hdf(self.data_path,'valid_out').to_numpy()

            if self.use_embedding:
                print('Using pre-trained embeddings!'+'-' * 60)
                self.train_x = pd.read_hdf(self.data_path,'train_in_3_mers')
                self.valid_x = pd.read_hdf(self.data_path,'valid_in_3_mers')
                total_length = self.train_x.shape[1]
                middle_index = total_length // 2
                self.train_x = self.train_x.iloc[:,middle_index-self.radius+1:middle_index+self.radius-1+1].to_numpy()
                # print(self.train_x.shape[1])
                self.valid_x = self.valid_x.iloc[:,middle_index-self.radius+1:middle_index+self.radius-1+1].to_numpy()
            else:
                # cropping the sequence one_hot encoding
                total_length = self.train_x.shape[1]
                middle_index = total_length // 2
                # print(middle_index)
                self.train_x = self.train_x.iloc[:,2000-self.radius*4:2004+self.radius*4].to_numpy()
                self.valid_x = self.valid_x.iloc[:,2000-self.radius*4:2004+self.radius*4].to_numpy()
        else:
            self.valid_x = pd.read_hdf(self.data_path,'valid_in')
            self.valid_y = pd.read_hdf(self.data_path,'valid_out').to_numpy()
            self.test_x = pd.read_hdf(self.data_path,'test_in')
            self.test_y = pd.read_hdf(self.data_path,'test_out').to_numpy()

            if self.use_embedding:
                self.valid_x = pd.read_hdf(self.data_path,'valid_in_3_mers')
                self.test_x = pd.read_hdf(self.data_path,'test_in_3_mers')
                # cropping the sequence accroding to its length
                total_length = self.valid_x.shape[1]
                middle_index = total_length // 2
                # print(self.train_x.shape[1])
                self.valid_x = self.valid_x.iloc[:,middle_index-self.radius+1:middle_index+self.radius-1+1].to_numpy()
                self.test_x = self.test_x.iloc[:,middle_index-self.radius+1:middle_index+self.radius-1+1].to_numpy()
            else:
                # cropping the sequence one_hot encoding
                total_length = self.valid_x.shape[1]
                middle_index = total_length // 2
                # print(middle_index)
                self.valid_x = self.valid_x.iloc[:,2000-self.radius*4:2004+self.radius*4].to_numpy()
                # print(self.train_x.shape[1])
                self.test_x = self.test_x.iloc[:,2000-self.radius*4:2004+self.radius*4].to_numpy()
"""
        new_rmdata = """        if self.mode == 'train':
            self.train_y = pd.read_hdf(self.data_path,'train_out').to_numpy()
            self.valid_y = pd.read_hdf(self.data_path,'valid_out').to_numpy()

            if self.use_embedding:
                print('Using pre-trained embeddings!'+'-' * 60)
                self.train_x = pd.read_hdf(self.data_path,'train_in_3_mers')
                self.valid_x = pd.read_hdf(self.data_path,'valid_in_3_mers')
                total_length = self.train_x.shape[1]
                middle_index = total_length // 2
                self.train_x = self.train_x.iloc[:,middle_index-self.radius+1:middle_index+self.radius-1+1].to_numpy()
                self.valid_x = self.valid_x.iloc[:,middle_index-self.radius+1:middle_index+self.radius-1+1].to_numpy()
            else:
                self.train_x = pd.read_hdf(self.data_path,'train_in')
                self.valid_x = pd.read_hdf(self.data_path,'valid_in')
                # cropping the sequence one_hot encoding
                total_length = self.train_x.shape[1]
                middle_index = total_length // 2
                self.train_x = self.train_x.iloc[:,2000-self.radius*4:2004+self.radius*4].to_numpy()
                self.valid_x = self.valid_x.iloc[:,2000-self.radius*4:2004+self.radius*4].to_numpy()
        else:
            self.valid_y = pd.read_hdf(self.data_path,'valid_out').to_numpy()
            self.test_y = pd.read_hdf(self.data_path,'test_out').to_numpy()

            if self.use_embedding:
                self.valid_x = pd.read_hdf(self.data_path,'valid_in_3_mers')
                self.test_x = pd.read_hdf(self.data_path,'test_in_3_mers')
                # cropping the sequence accroding to its length
                total_length = self.valid_x.shape[1]
                middle_index = total_length // 2
                self.valid_x = self.valid_x.iloc[:,middle_index-self.radius+1:middle_index+self.radius-1+1].to_numpy()
                self.test_x = self.test_x.iloc[:,middle_index-self.radius+1:middle_index+self.radius-1+1].to_numpy()
            else:
                self.valid_x = pd.read_hdf(self.data_path,'valid_in')
                self.test_x = pd.read_hdf(self.data_path,'test_in')
                # cropping the sequence one_hot encoding
                total_length = self.valid_x.shape[1]
                middle_index = total_length // 2
                self.valid_x = self.valid_x.iloc[:,2000-self.radius*4:2004+self.radius*4].to_numpy()
                self.test_x = self.test_x.iloc[:,2000-self.radius*4:2004+self.radius*4].to_numpy()
"""
        if old_rmdata in text:
            text = text.replace(old_rmdata, new_rmdata)
        if "def _safe_roc_auc_score(" not in text:
            marker = "def cal_metrics(model_out,label,plot=False,class_names=None,plot_name=None):\n"
            helper = """def _safe_roc_auc_score(y_true, y_score):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y_true)) < 2:
        return float('nan')
    return roc_auc_score(y_true, y_score)

def _safe_average_precision_score(y_true, y_score):
    from sklearn.metrics import average_precision_score
    if np.sum(y_true) == 0:
        return float('nan')
    return average_precision_score(y_true, y_score)

"""
            text = text.replace(marker, helper + marker)
        text = text.replace(
            "auc = roc_auc_score(y_true[i*100:(i+1)*100],y_score[i*100:(i+1)*100])",
            "auc = _safe_roc_auc_score(y_true[i*100:(i+1)*100],y_score[i*100:(i+1)*100])",
        )
        text = text.replace(
            "ap = average_precision_score(y_true[i*100:(i+1)*100],y_score[i*100:(i+1)*100])",
            "ap = _safe_average_precision_score(y_true[i*100:(i+1)*100],y_score[i*100:(i+1)*100])",
        )
        text = text.replace(
            "auc_2 = roc_auc_score(y_true,y_score)",
            "auc_2 = _safe_roc_auc_score(y_true,y_score)",
        )
        text = text.replace(
            "ap_2 = average_precision_score(y_true,y_score)",
            "ap_2 = _safe_average_precision_score(y_true,y_score)",
        )
        train_utils.write_text(text, encoding="utf-8")

    train_py = script_root / "train.py"
    if train_py.exists():
        text = train_py.read_text(encoding="utf-8", errors="replace")
        text = text.replace(
            "cal_metrics(y_pred, y_test, plot=True, class_names=class_names, plot_name=model_name)",
            "cal_metrics(y_pred, y_test, plot=False, class_names=class_names, plot_name=model_name)",
        )
        train_py.write_text(text, encoding="utf-8")

patch_optional_plotting(Path("${MULTIRM_DIR}") / "Scripts")
PY

echo "[6/8] Run official MultiRM train/test"
if [[ "${RUN_MULTIRM:-1}" == "1" ]]; then
  MULTIRM_RESULT="${WORK_ROOT_ABS}/multirm_official_length_${WINDOW_LENGTH}"
  mkdir -p "${MULTIRM_RESULT}"
  (
    cd "${MULTIRM_DIR}/Scripts"
    "${PYTHON_BIN}" train.py \
      --mode train \
      --inputs "${OFFICIAL_H5_ABS}" \
      --use_embedding True \
      --epochs "${OFFICIAL_EPOCHS}" \
      --batch_size "${MULTIRM_BATCH_SIZE}" \
      --length "${WINDOW_LENGTH}" \
      --lr 0.0001 \
      --lr_decay 0.8 \
      --gpu "${CUDA_ID}" \
      --num_task "${NUM_CLASSES}" \
      --save_dir "${MULTIRM_RESULT}" \
      2>&1 | tee "${WORK_ROOT_ABS}/multirm_train_${WINDOW_LENGTH}nt.log"
    "${PYTHON_BIN}" train.py \
      --mode test \
      --inputs "${OFFICIAL_H5_ABS}" \
      --use_embedding True \
      --batch_size "${MULTIRM_BATCH_SIZE}" \
      --length "${WINDOW_LENGTH}" \
      --weights "${MULTIRM_RESULT}/trained_model_${WINDOW_LENGTH}seqs.pkl" \
      --gpu "${CUDA_ID}" \
      --num_task "${NUM_CLASSES}" \
      --save_dir "${MULTIRM_RESULT}" \
      2>&1 | tee "${WORK_ROOT_ABS}/multirm_test_${WINDOW_LENGTH}nt.log"
  )
else
  echo "Skip MultiRM because RUN_MULTIRM=0"
fi

echo "[7/8] Prepare patched TransRNAm official code workspace"
TRANSRNAM_WORK="${WORK_ROOT_ABS}/TransRNAm-main"
if [[ ! -d "${TRANSRNAM_WORK}" ]]; then
  cp -a "${TRANSRNAM_SOURCE_DIR}" "${TRANSRNAM_WORK}"
fi
mkdir -p "${TRANSRNAM_WORK}/Embeddings" "${TRANSRNAM_WORK}/Results"
if [[ ! -s "${TRANSRNAM_WORK}/Embeddings/embeddings_12RM.pkl" ]]; then
  cp "${MULTIRM_DIR}/Embeddings/embeddings_12RM.pkl" "${TRANSRNAM_WORK}/Embeddings/embeddings_12RM.pkl"
fi
cp "${TRANSRNAM_JSON_DIR}"/*.json "${TRANSRNAM_WORK}/Scripts/"
"${PYTHON_BIN}" - <<PY
from pathlib import Path
root = Path("${TRANSRNAM_WORK}") / "Scripts"
for name in ["train.py", "models.py", "util_layers.py"]:
    path = root / name
    text = path.read_text(encoding="utf-8", errors="replace")
    text = text.replace("numgpu=3", "numgpu=0").replace("numgpu = 3", "numgpu = 0")
    if name == "train.py":
        text = text.replace(
            "cal_metrics(y_pred, y_test, plot=True, class_names=class_names, plot_name=model_name)",
            "cal_metrics(y_pred, y_test, plot=False, class_names=class_names, plot_name=model_name)",
        )
    path.write_text(text, encoding="utf-8")

train_utils = root / "train_utils.py"
if train_utils.exists():
    text = train_utils.read_text(encoding="utf-8", errors="replace")
    old = """import matplotlib.pyplot as plt
from itertools import cycle
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
"""
    new = """from itertools import cycle
try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import matplotlib as mpl
    from matplotlib.font_manager import FontProperties
except Exception:
    plt = None
    ListedColormap = None
    mpl = None
    FontProperties = None
"""
    if old in text:
        text = text.replace(old, new)
    old_rmdata = """        if self.mode == 'train':
            self.train_x = pd.read_hdf(self.data_path,'train_in')
            self.train_y = pd.read_hdf(self.data_path,'train_out').to_numpy()
            self.valid_x = pd.read_hdf(self.data_path,'valid_in')
            self.valid_y = pd.read_hdf(self.data_path,'valid_out').to_numpy()

            if self.use_embedding:
                print('Using pre-trained embeddings!'+'-' * 60)
                self.train_x = pd.read_hdf(self.data_path,'train_in_3_mers')
                self.valid_x = pd.read_hdf(self.data_path,'valid_in_3_mers')
                total_length = self.train_x.shape[1]
                middle_index = total_length // 2
                self.train_x = self.train_x.iloc[:,middle_index-self.radius+1:middle_index+self.radius-1+1].to_numpy()
                # print(self.train_x.shape[1])
                self.valid_x = self.valid_x.iloc[:,middle_index-self.radius+1:middle_index+self.radius-1+1].to_numpy()
            else:
                # cropping the sequence one_hot encoding
                total_length = self.train_x.shape[1]
                middle_index = total_length // 2
                # print(middle_index)
                self.train_x = self.train_x.iloc[:,2000-self.radius*4:2004+self.radius*4].to_numpy()
                self.valid_x = self.valid_x.iloc[:,2000-self.radius*4:2004+self.radius*4].to_numpy()
        else:
            self.valid_x = pd.read_hdf(self.data_path,'valid_in')
            self.valid_y = pd.read_hdf(self.data_path,'valid_out').to_numpy()
            self.test_x = pd.read_hdf(self.data_path,'test_in')
            self.test_y = pd.read_hdf(self.data_path,'test_out').to_numpy()

            if self.use_embedding:
                self.valid_x = pd.read_hdf(self.data_path,'valid_in_3_mers')
                self.test_x = pd.read_hdf(self.data_path,'test_in_3_mers')
                # cropping the sequence accroding to its length
                total_length = self.valid_x.shape[1]
                middle_index = total_length // 2
                # print(self.train_x.shape[1])
                self.valid_x = self.valid_x.iloc[:,middle_index-self.radius+1:middle_index+self.radius-1+1].to_numpy()
                self.test_x = self.test_x.iloc[:,middle_index-self.radius+1:middle_index+self.radius-1+1].to_numpy()
            else:
                # cropping the sequence one_hot encoding
                total_length = self.valid_x.shape[1]
                middle_index = total_length // 2
                # print(middle_index)
                self.valid_x = self.valid_x.iloc[:,2000-self.radius*4:2004+self.radius*4].to_numpy()
                # print(self.train_x.shape[1])
                self.test_x = self.test_x.iloc[:,2000-self.radius*4:2004+self.radius*4].to_numpy()
"""
    new_rmdata = """        if self.mode == 'train':
            self.train_y = pd.read_hdf(self.data_path,'train_out').to_numpy()
            self.valid_y = pd.read_hdf(self.data_path,'valid_out').to_numpy()

            if self.use_embedding:
                print('Using pre-trained embeddings!'+'-' * 60)
                self.train_x = pd.read_hdf(self.data_path,'train_in_3_mers')
                self.valid_x = pd.read_hdf(self.data_path,'valid_in_3_mers')
                total_length = self.train_x.shape[1]
                middle_index = total_length // 2
                self.train_x = self.train_x.iloc[:,middle_index-self.radius+1:middle_index+self.radius-1+1].to_numpy()
                self.valid_x = self.valid_x.iloc[:,middle_index-self.radius+1:middle_index+self.radius-1+1].to_numpy()
            else:
                self.train_x = pd.read_hdf(self.data_path,'train_in')
                self.valid_x = pd.read_hdf(self.data_path,'valid_in')
                # cropping the sequence one_hot encoding
                total_length = self.train_x.shape[1]
                middle_index = total_length // 2
                self.train_x = self.train_x.iloc[:,2000-self.radius*4:2004+self.radius*4].to_numpy()
                self.valid_x = self.valid_x.iloc[:,2000-self.radius*4:2004+self.radius*4].to_numpy()
        else:
            self.valid_y = pd.read_hdf(self.data_path,'valid_out').to_numpy()
            self.test_y = pd.read_hdf(self.data_path,'test_out').to_numpy()

            if self.use_embedding:
                self.valid_x = pd.read_hdf(self.data_path,'valid_in_3_mers')
                self.test_x = pd.read_hdf(self.data_path,'test_in_3_mers')
                # cropping the sequence accroding to its length
                total_length = self.valid_x.shape[1]
                middle_index = total_length // 2
                self.valid_x = self.valid_x.iloc[:,middle_index-self.radius+1:middle_index+self.radius-1+1].to_numpy()
                self.test_x = self.test_x.iloc[:,middle_index-self.radius+1:middle_index+self.radius-1+1].to_numpy()
            else:
                self.valid_x = pd.read_hdf(self.data_path,'valid_in')
                self.test_x = pd.read_hdf(self.data_path,'test_in')
                # cropping the sequence one_hot encoding
                total_length = self.valid_x.shape[1]
                middle_index = total_length // 2
                self.valid_x = self.valid_x.iloc[:,2000-self.radius*4:2004+self.radius*4].to_numpy()
                self.test_x = self.test_x.iloc[:,2000-self.radius*4:2004+self.radius*4].to_numpy()
"""
    if old_rmdata in text:
        text = text.replace(old_rmdata, new_rmdata)
    if "def _safe_roc_auc_score(" not in text:
        marker = "def cal_metrics(model_out,label,plot=False,class_names=None,plot_name=None):\n"
        helper = """def _safe_roc_auc_score(y_true, y_score):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y_true)) < 2:
        return float('nan')
    return roc_auc_score(y_true, y_score)

def _safe_average_precision_score(y_true, y_score):
    from sklearn.metrics import average_precision_score
    if np.sum(y_true) == 0:
        return float('nan')
    return average_precision_score(y_true, y_score)

"""
        text = text.replace(marker, helper + marker)
    text = text.replace(
        "auc = roc_auc_score(y_true[i*100:(i+1)*100],y_score[i*100:(i+1)*100])",
        "auc = _safe_roc_auc_score(y_true[i*100:(i+1)*100],y_score[i*100:(i+1)*100])",
    )
    text = text.replace(
        "ap = average_precision_score(y_true[i*100:(i+1)*100],y_score[i*100:(i+1)*100])",
        "ap = _safe_average_precision_score(y_true[i*100:(i+1)*100],y_score[i*100:(i+1)*100])",
    )
    text = text.replace(
        "auc_2 = roc_auc_score(y_true,y_score)",
        "auc_2 = _safe_roc_auc_score(y_true,y_score)",
    )
    text = text.replace(
        "ap_2 = average_precision_score(y_true,y_score)",
        "ap_2 = _safe_average_precision_score(y_true,y_score)",
    )
    train_utils.write_text(text, encoding="utf-8")
PY

echo "[8/8] Run official TransRNAm train/test"
if [[ "${RUN_TRANSRNAM:-1}" == "1" ]]; then
  TRANSRNAM_RESULT="${WORK_ROOT_ABS}/transrnam_official_length_${WINDOW_LENGTH}"
  mkdir -p "${TRANSRNAM_RESULT}"
  (
    cd "${TRANSRNAM_WORK}/Scripts"
    "${PYTHON_BIN}" train.py \
      --mode train \
      --inputs "${OFFICIAL_H5_ABS}" \
      --use_embedding True \
      --epochs "${OFFICIAL_EPOCHS}" \
      --batch_size "${TRANSRNAM_BATCH_SIZE}" \
      --length "${WINDOW_LENGTH}" \
      --lr 0.0001 \
      --lr_decay 0.95 \
      --gpu "${CUDA_ID}" \
      --num_task "${NUM_CLASSES}" \
      --save_dir "${TRANSRNAM_RESULT}" \
      2>&1 | tee "${WORK_ROOT_ABS}/transrnam_train_${WINDOW_LENGTH}nt.log"
    "${PYTHON_BIN}" train.py \
      --mode test \
      --inputs "${OFFICIAL_H5_ABS}" \
      --use_embedding True \
      --batch_size "${TRANSRNAM_BATCH_SIZE}" \
      --length "${WINDOW_LENGTH}" \
      --weights "${TRANSRNAM_RESULT}/result.pkl" \
      --gpu "${CUDA_ID}" \
      --num_task "${NUM_CLASSES}" \
      --save_dir "${TRANSRNAM_RESULT}" \
      2>&1 | tee "${WORK_ROOT_ABS}/transrnam_test_${WINDOW_LENGTH}nt.log"
  )
else
  echo "Skip TransRNAm because RUN_TRANSRNAM=0"
fi

cat <<EOF
Done.

Official-compatible input:
  ${OFFICIAL_H5}
  ${TRANSRNAM_JSON_DIR}

Logs/results:
  ${WORK_ROOT}

Important:
  Official scripts write their own pf.json/log.csv. For paper tables, re-evaluate
  saved predictions or rerun the portable evaluator so metrics stay comparable.
EOF
