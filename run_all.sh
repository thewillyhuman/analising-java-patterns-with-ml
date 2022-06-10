# Download full tables.
python build_dataset.py --database-name patternmining --data-dir data/base/programs --out-file-name programs_full.csv;
python build_dataset.py --database-name patternmining --data-dir data/base/type_defs --out-file-name type_defs_full.csv;
python build_dataset.py --database-name patternmining --data-dir data/base/field_defs --out-file-name field_defs_full.csv;
python build_dataset.py --database-name patternmining --data-dir data/base/method_defs --out-file-name method_defs_full.csv;
python build_dataset.py --database-name patternmining --data-dir data/base/types --out-file-name types_full.csv;
python build_dataset.py --database-name patternmining --data-dir data/base/statements --out-file-name statements_full.csv;
python build_dataset.py --database-name patternmining --data-dir data/base/expressions --out-file-name expressions_full.csv;
python build_dataset.py --database-name patternmining --data-dir data/base/het1 --out-file-name het1_full.csv;
python build_dataset.py --database-name patternmining --data-dir data/base/het2 --out-file-name het2_full.csv;
python build_dataset.py --database-name patternmining --data-dir data/base/het3 --out-file-name het3_full.csv;
python build_dataset.py --database-name patternmining --data-dir data/base/het4 --out-file-name het4_full.csv;
python build_dataset.py --database-name patternmining --data-dir data/base/het5 --out-file-name het5_full.csv;

# Download unique tables.
python build_dataset.py --database-name patternminingV2 --data-dir data/unique/programs --out-file-name programs_unique.csv;
python build_dataset.py --database-name patternminingV2 --data-dir data/unique/type_defs --out-file-name type_defs_unique.csv;
python build_dataset.py --database-name patternminingV2 --data-dir data/unique/field_defs --out-file-name field_defs_unique.csv;
python build_dataset.py --database-name patternminingV2 --data-dir data/unique/method_defs --out-file-name method_defs_unique.csv;
python build_dataset.py --database-name patternminingV2 --data-dir data/unique/types --out-file-name types_unique.csv;
python build_dataset.py --database-name patternminingV2 --data-dir data/unique/statements --out-file-name statements_unique.csv;
python build_dataset.py --database-name patternminingV2 --data-dir data/unique/expressions --out-file-name expressions_unique.csv;
python build_dataset.py --database-name patternminingV2 --data-dir data/unique/het1 --out-file-name het1_unique.csv;
python build_dataset.py --database-name patternminingV2 --data-dir data/unique/het2 --out-file-name het2_unique.csv;
python build_dataset.py --database-name patternminingV2 --data-dir data/unique/het3 --out-file-name het3_unique.csv;
python build_dataset.py --database-name patternminingV2 --data-dir data/unique/het4 --out-file-name het4_unique.csv;
python build_dataset.py --database-name patternminingV2 --data-dir data/unique/het5 --out-file-name het5_unique.csv;

# Run all log regs with full tables
python logreg_train.py --data-dir data/base/programs --model-dir experiments/logreg/base/programs/base_model --database-name patternmining;
python logreg_train.py --data-dir data/base/type_defs --model-dir experiments/logreg/base/type_defs/base_model --database-name patternmining;
python logreg_train.py --data-dir data/base/field_defs --model-dir experiments/logreg/base/field_defs/base_model --database-name patternmining;
python logreg_train.py --data-dir data/base/method_defs --model-dir experiments/logreg/base/method_defs/base_model --database-name patternmining;
python logreg_train.py --data-dir data/base/types --model-dir experiments/logreg/base/types/base_model --database-name patternmining;
python logreg_train.py --data-dir data/base/statements --model-dir experiments/logreg/base/statements/base_model --database-name patternmining;
python logreg_train.py --data-dir data/base/expressions --model-dir experiments/logreg/base/expressions/base_model --database-name patternmining;
python logreg_train.py --data-dir data/base/het1 --model-dir experiments/logreg/base/het1/base_model --database-name patternmining;
python logreg_train.py --data-dir data/base/het2 --model-dir experiments/logreg/base/het2/base_model --database-name patternmining;
python logreg_train.py --data-dir data/base/het3 --model-dir experiments/logreg/base/het3/base_model --database-name patternmining;
python logreg_train.py --data-dir data/base/het4 --model-dir experiments/logreg/base/het4/base_model --database-name patternmining;
python logreg_train.py --data-dir data/base/het5 --model-dir experiments/logreg/base/het5/base_model --database-name patternmining;

# Run all log regs with unique tables
python logreg_train.py --data-dir data/unique/programs --model-dir experiments/logreg/unique/programs/base_model --database-name patternminingV2;
python logreg_train.py --data-dir data/unique/type_defs --model-dir experiments/logreg/unique/type_defs/base_model --database-name patternminingV2;
python logreg_train.py --data-dir data/unique/field_defs --model-dir experiments/logreg/unique/field_defs/base_model --database-name patternminingV2;
python logreg_train.py --data-dir data/unique/method_defs --model-dir experiments/logreg/unique/method_defs/base_model --database-name patternminingV2;
python logreg_train.py --data-dir data/unique/types --model-dir experiments/logreg/unique/types/base_model --database-name patternminingV2;
python logreg_train.py --data-dir data/unique/statements --model-dir experiments/logreg/unique/statements/base_model --database-name patternminingV2;
python logreg_train.py --data-dir data/unique/expressions --model-dir experiments/logreg/unique/expressions/base_model --database-name patternminingV2;
python logreg_train.py --data-dir data/unique/het1 --model-dir experiments/logreg/unique/het1/base_model --database-name patternminingV2;
python logreg_train.py --data-dir data/unique/het2 --model-dir experiments/logreg/unique/het2/base_model --database-name patternminingV2;
python logreg_train.py --data-dir data/unique/het3 --model-dir experiments/logreg/unique/het3/base_model --database-name patternminingV2;
python logreg_train.py --data-dir data/unique/het4 --model-dir experiments/logreg/unique/het4/base_model --database-name patternminingV2;
python logreg_train.py --data-dir data/unique/het5 --model-dir experiments/logreg/unique/het5/base_model --database-name patternminingV2;