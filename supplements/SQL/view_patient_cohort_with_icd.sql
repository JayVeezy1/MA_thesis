-- SET search_path TO mimiciii;

create or replace view patient_cohort_with_icd
AS
	with diagnoses as
	(
		select * from get_all_diagnoses()
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
			diag.icd9_code,
			diag.all_icd9_codes
		from mimiciii.patient_cohort_view cohort
		inner join diagnoses diag
			on cohort.hadm_id = diag.hadm_id
	)
	select
		*
	from t1 
	where excluded = 0					-- remove all 'excluded' entries
	order by t1.hadm_id, t1.seq_num;
end;

-- select * from patient_cohort_with_icd;

-- select count(*) from  patient_cohort_with_icd;
-- join leads to 705921 entries, but in diagnoses only 600.000 hadm_id entries 
-- because in cohort_view some hadm_ids are duplicate, because it was 1 hadm but multiple times icustay_id
-- solution: with "excluded = 1" count = 513035 and independent on join-type (right, left, inner) -> always same matches.

	