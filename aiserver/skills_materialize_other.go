//go:build !unix

package aiserver

// supportsDirFD is false where there is no *at() family. Materialization
// refuses to run rather than fall back to path-based checks: Windows is not a
// supported platform for this release.
const supportsDirFD = false

func writeSkills(sel selection, root string, opts *WriteSkillsOptions) (ReconcileReport, error) {
	if _, err := parseWriteOptions(opts); err != nil {
		return ReconcileReport{}, err
	}
	return ReconcileReport{}, ErrUnsupportedPlatform
}
