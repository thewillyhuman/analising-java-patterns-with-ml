SELECT program_id as program__program_id,
       classpercentage as program__percentage_of_classes,
       interfacepercentage as program__percentage_of_interfaces,
       enumpercentage as program__percentage_of_enums,
       codeinpackages as program__contains_code_in_packages,
       defaultpackage as program__contains_code_in_default_package,
       percentage_of_types_in_packages as program__percentage_of_types_in_packages,
       percentage_of_types_default_package as program__percentage_of_types_in_default_package,
       user_id as program__user_id,
       user_class as program__user_class
FROM PROGRAM;