#!/usr/bin/env bash

readonly USAGE="\nUsage: bash pipeline.sh --train SDF1 [SDF2 SDF3 ...] [--test SDF1 [SDF2 SDF3 ...]]
  [--grouping SDF1 [SDF2 SDF3 ...]] [--quick-run KEY] [--exp-tol FLOAT] [--max-err FLOAT]\n\n"

readonly KEYS=(R1_oV R3_oV R1_V-R3 R1_V-R4 R1_V-R5 R1_V-R6)
readonly KEYS_TEXT="Possible keys for quick run to locate titratable groups:
  R1_oV   - Radius 1 only
  R3_oV   - Radius 3 only
  R1_V-R3 - Radius 1 search with radius 3 validation
  R1_V-R4 - Radius 1 search with radius 4 validation
  R1_V-R5 - Radius 1 search with radius 5 validation
  R1_V-R6 - Radius 1 search with radius 6 validation\n\n"

function print_help {
    printf "$USAGE"
    if [ $# -ne 0 ]; then
      printf "$KEYS_TEXT"
    fi
}

if [ $# -eq 0 ] || [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
  print_help full
  exit 0
fi

if [ -z "$PKA_CODEBASE" ]; then
  printf "\nPlease set the environment variable \"PKA_CODEBASE\" to the repository root path.\ne.g. \"export PKA_CODEBASE=/path/to/repo\"\n\n"
  exit 7
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PKA_CODEBASE:$PYTHONPATH"

quick_run=false
train_files=()
test_files=()
grouping_files=()
train_file="combined_training_datasets.sdf"
grouping_file="combined_grouping_datasets.sdf"
ddl_file="grouping_smiles.smi"
atom_col_to_use="R1_V-R5"
exp_tolerance="0.3"
max_error_for_value_assignment="2"

while (($#)); do
  case $1 in
    "--train")
      shift
      while [[ $1 != --* ]] && [ $# -ne 0 ]; do
        train_files+=("$1")
        shift
      done
      if [ ${#train_files[@]} -eq 0 ]; then
        printf "\nInvalid call!"
        print_help
        exit 1
      fi
    ;;
    "--test")
      shift
      while [[ $1 != --* ]] && [ $# -ne 0 ]; do
        test_files+=("$1")
        shift
      done
      if [ ${#test_files[@]} -eq 0 ]; then
        printf "\nInvalid call!"
        print_help
        exit 1
      fi
    ;;
    "--grouping")
      shift
      while [[ $1 != --* ]] && [ $# -ne 0 ]; do
        grouping_files+=("$1")
        shift
      done
      if [ ${#grouping_files[@]} -eq 0 ]; then
        printf "\nInvalid call!"
        print_help
        exit 1
      fi
    ;;
    "--quick-run")
      quick_run=true
      shift
      # shellcheck disable=SC2076
      if [[ ! " ${KEYS[*]} " =~ " $1 " ]]; then
        printf "\nInvalid key for quick run!"
        print_help full
        exit 1
      fi
      atom_col_to_use="$1"
      shift
    ;;
    "--exp-tol")
      shift
      if [[ ! $1 =~ ^[0-9]+\.?[0-9]*$ ]]; then
        printf "\n--exp-tol needs a positive float value!"
        print_help
        exit 1
      fi
      exp_tolerance="$1"
      shift
    ;;
    "--max-err")
      shift
      if [[ ! $1 =~ ^[0-9]+\.?[0-9]*$ ]]; then
        printf "\n--max-err needs a positive float value!"
        print_help
        exit 1
      fi
      max_error_for_value_assignment="$1"
      shift
    ;;
    *)
      printf "\nInvalid call!"
      print_help
      exit 1
    ;;
  esac
done

if [ "$quick_run" = true ]; then
  radius=${atom_col_to_use:1:1}
  if [[ $atom_col_to_use =~ "V-R" ]]; then
    tree=${atom_col_to_use:6:1}
  else
    tree="-1"
  fi
fi

echo
echo "Combining training datasets..."
python "$PKA_CODEBASE/pipeline/combine_datasets.py" "${train_files[@]}" > "$train_file" || exit 3

echo
if [ ${#grouping_files[@]} -ne 0 ]; then
  echo "Combining grouping datasets..."
  python "$PKA_CODEBASE/pipeline/combine_datasets.py" "${grouping_files[@]}" > "$grouping_file" || exit 8
else
  echo "Combining grouping datasets using all specified train and test datasets..."
  python "$PKA_CODEBASE/pipeline/combine_datasets.py" "${train_files[@]}" "${test_files[@]}" > "$grouping_file" || exit 8
fi

cu_test_files=()
for sdf in "${test_files[@]}" "$train_file" "$grouping_file"; do
  echo
  echo "Cleaning and filtering ($(basename "$sdf"))..."
  cm_filename="$(basename "$sdf" .sdf)_cleaned_unique.sdf"
  python "$PKA_CODEBASE/pipeline/gen_clean_unique_dataset.py" "$sdf" "$cm_filename" || exit 2
  if [ "$sdf" == "$train_file" ]; then
    train_file="$cm_filename"
  elif [ "$sdf" == "$grouping_file" ]; then
    grouping_file="$cm_filename"
  else
    cu_test_files+=("$cm_filename")
  fi
done

cu_test_files_no_td=()
if [ ${#cu_test_files[@]} -ne 0 ]; then
  echo
  echo "Remove training data from test files..."
  python "$PKA_CODEBASE/pipeline/remove_traindata_from_testdata.py" "$train_file" "${cu_test_files[@]}" || exit 6
  for sdf in "${cu_test_files[@]}"; do
    cm_filename="$(basename "$sdf" .sdf)_notraindata.sdf"
    cu_test_files_no_td+=("$cm_filename")
  done
fi

echo
echo "Converting grouping SDF to SMI file..."
python "$PKA_CODEBASE/pipeline/sdf_to_smi.py" "$grouping_file" "$ddl_file" || exit 11

echo
echo "Running Dimporphite-DL (see dimorphite_dl.log)..."
if [ ! -f "$PKA_CODEBASE/dimorphite_dl/dimorphite_dl.py" ]; then
  echo "Fetching dimorphite_dl submodule..."
  cd "$PKA_CODEBASE/dimorphite_dl" || exit 14
  git submodule update --init --recursive || exit 14
  cd - || exit 14
fi
ddl_out_file="protonated_$ddl_file"
python "$PKA_CODEBASE/dimorphite_dl/dimorphite_dl.py" --min_ph 2 --max_ph 12 \
  --smiles_file "$ddl_file" --output_file "$ddl_out_file" --max_variants 100000 > dimorphite_dl.log 2>&1 || exit 9

echo
echo "Generating SMARTS tree from grouping data..."
if [ "$quick_run" = true ]; then
  python "$PKA_CODEBASE/pipeline/gen_smarts_tree.py" "$grouping_file" "$ddl_out_file" "$radius" "$tree" || exit 10
else
  python "$PKA_CODEBASE/pipeline/gen_smarts_tree.py" "$grouping_file" "$ddl_out_file" || exit 10
fi

if [ "$quick_run" = false ]; then
  echo
  echo "Validate localization concepts..."
  python "$PKA_CODEBASE/pipeline/validate_results.py" "$grouping_file" "titratable_groups_SMARTS_R1.csv" || exit 12
fi

echo
echo "Locate titratable groups for training dataset and test datasets..."
for sdf in "${cu_test_files_no_td[@]}" "$train_file"; do
  python "$PKA_CODEBASE/pipeline/validate_results.py" "$sdf" "titratable_groups_SMARTS_R1.csv" "$atom_col_to_use" || exit 15
done

if [ "$quick_run" = false ]; then
  echo
  echo "Validate group assignment..."
  python "$PKA_CODEBASE/pipeline/values_to_groups.py" "$atom_col_to_use" "$exp_tolerance" "$max_error_for_value_assignment" df_with_loc.pkl 0 || exit 13
fi

echo
echo "Assign values to titratable groups for training dataset and test datasets..."
for sdf in "${cu_test_files_no_td[@]}" "$train_file"; do
  python "$PKA_CODEBASE/pipeline/values_to_groups.py" "$atom_col_to_use" "$exp_tolerance" "$max_error_for_value_assignment" "df_with_loc_$(basename "$sdf" .sdf).pkl" 1 || exit 13
done
