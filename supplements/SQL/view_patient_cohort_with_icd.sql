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
			cohort.los,
			cohort.icustays_count,
			cohort.age,
			cohort.patientweight,
			cohort.dob,
			cohort.dod,
			cohort.death_in_hosp,
			cohort.death_3_days,
			cohort.death_30_days,
			cohort.death_365_days,
			cohort.gender,
			cohort.ethnicity,
			cohort.first_service,
			cohort.dbsource,
			cohort.exclusion_secondarystay,
			cohort.exclusion_nonadult,
			-- cohort.exclusion_csurg,
			cohort.exclusion_carevue,
			cohort.exclusion_bad_data,
			cohort.excluded,
			diag.subject_id,
			diag.seq_num,
			diag.icd9_code,
			diag.all_icd9_codes
		from patient_cohort_view cohort
		inner join diagnoses diag
			on cohort.hadm_id = diag.hadm_id
	)
	select
		*
	from t1 
	where excluded = 0					-- remove all 'excluded' entries
	order by t1.hadm_id, t1.seq_num;
end;

-- select count(*) from  patient_cohort_with_icd;
-- join leads to 705921 entries, but in diagnoses only 600.000 hadm_id entries 
-- because in cohort_view some hadm_ids are duplicate, because it was 1 hadm but multiple times icustay_id
-- solution: with "excluded = 0" count = 513035 and independent on join-type (right, left, inner) -> always same matches.

	