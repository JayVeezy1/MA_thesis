-- SET search_path TO mimiciii;

create or replace view patient_cohort_with_icd
AS
	with diagnoses as
	(
		select 
			row_id,
			subject_id,
			diagnoses_icd.hadm_id,
			seq_num,
			icd9_code
		/*	
		, case 
				when left(diag.icd9_code, 3) < 100 then 'first'
			 	when left(diag.icd9_code, 3) < 100 then 'first'
			 	when left(diag.icd9_code, 3) < 100 then 'first'
		  		else 'no_category' 
			end as icd_category
		*/
		from mimiciii.diagnoses_icd
	), t1 as
	(
		select 
			cohort.hadm_id,
			cohort.icustay_id,
			cohort.intime,
			cohort.outtime,
			cohort.age,
			cohort.gender,
			cohort.ethnicity,
			cohort.first_service,
			cohort.dbsource,
			cohort.exclusion_secondarystay,
			cohort.exclusion_nonadult,
			cohort.exclusion_csurg,
			cohort.exclusion_carevue,
			cohort.exclusion_bad_data,
			cohort.excluded,
			diag.subject_id,
			diag.seq_num,
			diag.icd9_code
		from mimiciii.patient_cohort_view cohort
		inner join diagnoses diag
			on cohort.hadm_id = diag.hadm_id
	)
	select
		*
	from t1 
	where excluded = 0		-- TODO: check&rework the excluded logic (read paper)
	order by t1.icustay_id;

end;

select * from patient_cohort_with_icd;