#!/usr/bin/env bash

readonly USAGE="\nUsage: %s --train SDF1 [SDF2 SDF3 ...] [--test SDF1 [SDF2 SDF3 ...]] [--grouping SDF1 [SDF2 SDF3 ...]]\n\n"

if [ $# -eq 0 ] || [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
  printf "$USAGE" "$0"
  exit 0
fi

if [ -z "$PKA_CODEBASE" ]; then
  printf "\nPlease set the environment variable \"PKA_CODEBASE\" to the repository root path.\ne.g. \"export PKA_CODEBASE=/path/to/repo\"\n\n"
  exit 7
fi

export PYTHONUNBUFFERED=1

train=0
test=0
grouping=0
train_files=()
test_files=()
grouping_files=()
train_file="combined_training_datasets.sdf"
grouping_file="combined_grouping_datasets.sdf"
ddl_file="grouping_smiles.smi"
atom_col_to_use="sma_atom_ids_vr5"
exp_tolerance="0.3"
max_error_for_value_assignment="2"

for sdf in "$@"; do
  case $sdf in
    "--train")
      train=1
      test=0
      grouping=0
    ;;
    "--test")
      train=0
      test=1
      grouping=0
    ;;
    "--grouping")
      train=0
      test=0
      grouping=1
    ;;
    *)
      if [ $train -eq 0 ] && [ $test -eq 0 ] && [ $grouping -eq 0 ]; then
        printf "\nInvalid call!"
        printf "$USAGE" "$0"
        exit 1
      fi
      if [ $train -eq 1 ]; then
        train_files+=("$sdf")
      elif [ $test -eq 1 ]; then
        test_files+=("$sdf")
      else
        grouping_files+=("$sdf")
      fi
    ;;
  esac
done

if [ ${#train_files[@]} -eq 0 ]; then
  printf "\nInvalid call!"
  printf "$USAGE" "$0"
  exit 1
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
  python "$PKA_CODEBASE/pipeline/gen_clean_unique_multi_dataset.py" "$sdf" "$cm_filename" || exit 2
  if [ "$sdf" == "$train_file" ]; then
    train_file="$cm_filename"
  elif [ "$sdf" == "$grouping_file" ]; then
    grouping_file="$cm_filename"
  else
    cu_test_files+=("$cm_filename")
  fi
done

if [ ${#cu_test_files[@]} -ne 0 ]; then
  echo
  echo "Remove training data from test files..."
  python "$PKA_CODEBASE/pipeline/remove_traindata_from_testdata.py" "$train_file" "${cu_test_files[@]}" || exit 6
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
python "$PKA_CODEBASE/pipeline/gen_smarts_tree.py" "$grouping_file" "$ddl_out_file" || exit 10

echo
echo "Validate localization concepts..."
python "$PKA_CODEBASE/pipeline/validate_results.py" "$grouping_file" "titratable_groups_SMARTS_R1.csv" || exit 12

echo
echo "Validate group assignment..."
python "$PKA_CODEBASE/pipeline/values_to_groups.py" "$atom_col_to_use" "$exp_tolerance" "$max_error_for_value_assignment" || exit 13

# TODO combine values and groups individually for each dataset
