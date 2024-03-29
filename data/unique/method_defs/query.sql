SELECT methodvisibility      as method_def__visibility,
       defaultimplementation as method_def__is_default_impl,
       isfinal               as method_def__is_final,
       hasoverride           as method_def__is_override,
       isstatic              as method_def__is_static,
       numberparameters      as method_def__number_of_parameters,
       numbergenerictypes    as method_def__number_of_generics,
       numberthrows          as method_def__number_of_throws,
       returntype            as method_def__return_type,
       numberannotations     as method_def__number_of_annotations,
       numberstmts           as method_def__number_of_statements,
       methodlocalvars       as method_def__number_of_local_vars,
       locvarnaming          as method_def__naming_convention_of_local_vars,
       namingconvention      as method_def__naming_convention,
       isabstract            as method_def__is_abstract,
       numberinnerclasses    as method_def__number_of_inner_classes,
       isconstructor         as method_def__is_constructor,
       isstrictfp            as method_def__is_strictfp,
       isnative              as method_def__is_native,
       issynchronized        as method_def__is_synchronized,
       numberofoverloaded    as method_def__number_of_times_overloaded,
       firstparametertype    as method_def__type_of_first_parameter,
       secondparametertype   as method_def__type_of_second_parameter,
       thirdparametertype    as method_def__type_of_third_parameter,
       user_class            as method_def__user_class

FROM methoddefinition;