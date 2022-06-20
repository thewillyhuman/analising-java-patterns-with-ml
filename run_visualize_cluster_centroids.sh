python visualize_cluster_centroids.py --data-dir data/base/type_defs \
																			--model-dir experiments/kmeans/base/type_defs/best_model \
																			--features type_def__number_of_annotations \
																									type_def__number_of_implements \
																									type_def__number_of_generics \
																									type_def__number_of_methods \
																									type_def__percentage_overloaded_methods \
																									type_def__number_of_constructors \
																									type_def__number_of_fields \
																									type_def__number_of_nested_types \
																									type_def__number_of_inner_types \
																									type_def__percentage_of_static_fields \
																									type_def__is_strictfp \
																									type_def__percentage_of_static_methods \
																			--features-titles "Number of Annotations" \
																			                  "Number of Implements" \
																			                  "Number of Generics" \
																			                  "Number of Methods" \
																			                  "Percentage of Overloaded Methods" \
																			                  "Number of Constructors" \
																			                  "Number of Fields" \
																			                  "Number of Nested Types" \
																			                  "Number of Inner Types" \
																			                  "Percentage of Static Fields" \
																			                  "Declared as strictfp" \
																			                  "Percentage of Static Methods" \
																			--vertical-charts 3 \
																			--horizontal-charts 4 \
																			--out-dir /Users/thewillyhuman/Desktop