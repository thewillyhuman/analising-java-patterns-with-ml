# Download full tables.
python build_dataset.py --database-name patternmining --data-dir data/base/programs;
python build_dataset.py --database-name patternmining --data-dir data/base/type_defs;
python build_dataset.py --database-name patternmining --data-dir data/base/field_defs;
python build_dataset.py --database-name patternmining --data-dir data/base/method_defs;
python build_dataset.py --database-name patternmining --data-dir data/base/types;
python build_dataset.py --database-name patternmining --data-dir data/base/statements;
python build_dataset.py --database-name patternmining --data-dir data/base/expressions;
python build_dataset.py --database-name patternmining --data-dir data/base/het1;
python build_dataset.py --database-name patternmining --data-dir data/base/het2;
python build_dataset.py --database-name patternmining --data-dir data/base/het3;
python build_dataset.py --database-name patternmining --data-dir data/base/het4;
python build_dataset.py --database-name patternmining --data-dir data/base/het5;

# Download unique tables.
python build_dataset.py --database-name patternminingV2 --data-dir data/unique/programs;
python build_dataset.py --database-name patternminingV2 --data-dir data/unique/type_defs;
python build_dataset.py --database-name patternminingV2 --data-dir data/unique/field_defs;
python build_dataset.py --database-name patternminingV2 --data-dir data/unique/method_defs;
python build_dataset.py --database-name patternminingV2 --data-dir data/unique/types;
python build_dataset.py --database-name patternminingV2 --data-dir data/unique/statements;
python build_dataset.py --database-name patternminingV2 --data-dir data/unique/expressions;
python build_dataset.py --database-name patternminingV2 --data-dir data/unique/het1;
python build_dataset.py --database-name patternminingV2 --data-dir data/unique/het2;
python build_dataset.py --database-name patternminingV2 --data-dir data/unique/het3;
python build_dataset.py --database-name patternminingV2 --data-dir data/unique/het4;
python build_dataset.py --database-name patternminingV2 --data-dir data/unique/het5;

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

# Run all kmeans with full tables
python kmeans_fit.py --data-dir data/base/programs --model-dir experiments/kmeans/base/programs/best_model --compute-elbow --database-name patternmining;
python kmeans_fit.py --data-dir data/base/type_defs --model-dir experiments/kmeans/base/type_defs/best_model --compute-elbow --database-name patternmining;
python kmeans_fit.py --data-dir data/base/field_defs --model-dir experiments/kmeans/base/field_defs/best_model --compute-elbow --database-name patternmining;
python kmeans_fit.py --data-dir data/base/method_defs --model-dir experiments/kmeans/base/method_defs/best_model --compute-elbow --database-name patternmining;
python kmeans_fit.py --data-dir data/base/types --model-dir experiments/kmeans/base/types/best_model --compute-elbow --database-name patternmining;
python kmeans_fit.py --data-dir data/base/expressions --model-dir experiments/kmeans/base/expressions/best_model --compute-elbow --database-name patternmining;
python kmeans_fit.py --data-dir data/base/statements --model-dir experiments/kmeans/base/statements/best_model --compute-elbow --database-name patternmining;
python kmeans_fit.py --data-dir data/base/het1 --model-dir experiments/kmeans/base/het1/best_model --compute-elbow --database-name patternmining;
python kmeans_fit.py --data-dir data/base/het2 --model-dir experiments/kmeans/base/het2/best_model --compute-elbow --database-name patternmining;
python kmeans_fit.py --data-dir data/base/het3 --model-dir experiments/kmeans/base/het3/best_model --compute-elbow --database-name patternmining;
python kmeans_fit.py --data-dir data/base/het4 --model-dir experiments/kmeans/base/het4/best_model --compute-elbow --database-name patternmining;
python kmeans_fit.py --data-dir data/base/het5 --model-dir experiments/kmeans/base/het5/best_model --compute-elbow --database-name patternmining;