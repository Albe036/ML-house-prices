class MissingCorrelation(MissingHandling):
    def mannWhitneyU(
        self, featureMissingValues="", featuresReference=[], onlyTrue=True
    ):
        res = []
        missing_bin = self.dataFrame[featureMissingValues].isna().astype(int)
        cols = super()._filter_types_features(
            custom_features=featuresReference, type_features="numerical"
        )
        for col in cols:
            missing, present, GREAT_ENOUGH = self._filter_missing_and_present(
                missing_bin, col
            )
            if GREAT_ENOUGH:
                stat, p_value = mannwhitneyu(missing, present, alternative="two-sided")
                res.append(
                    {
                        "feature": col,
                        "stat": stat,
                        "p_value": p_value,
                        "evidence_MAR": (p_value < self.alpha),
                    }
                )
        return super()._config_output(res)