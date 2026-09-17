//! Prediction styles (M5): address form (German du/Sie), length policy,
//! and the static [`StyleSpec`] / resolved [`ResolvedStyle`] types.
//!
//! Detection and bans follow simple token rules from the plan: du-family
//! tokens are matched case-insensitively, Sie-family only capitalized
//! (lowercase `sie`/`ihr` are ambiguous and never banned).

/// Language tag for a style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LanguageSpec {
    /// Detect per request (interim: function-word heuristic).
    #[default]
    Auto,
    /// English base model.
    En,
    /// German base model.
    De,
}

/// German address form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AddressForm {
    /// Informal `du`.
    Du,
    /// Formal `Sie`.
    Sie,
    /// No constraint.
    #[default]
    None,
}

/// Suggestion length policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LengthMode {
    /// Words only (no sentence tier).
    Word,
    /// Stop at clause end (`, ; :` or sentence end).
    Phrase,
    /// Stop at sentence end (`. ! ?`).
    #[default]
    Sentence,
}

/// Static style specification (one `[styles.<id>]` TOML section in the
/// daemon config, plus built-ins below).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StyleSpec {
    /// Style id (e.g. `"default"`, `"formal-de"`).
    pub id: String,
    /// Language selection.
    pub language: LanguageSpec,
    /// Address-form constraint.
    pub address_form: AddressForm,
    /// Length policy.
    pub length: LengthMode,
}

impl StyleSpec {
    /// The default style: auto language, no address constraint, sentences.
    pub fn default_style() -> Self {
        Self {
            id: "default".to_string(),
            language: LanguageSpec::Auto,
            address_form: AddressForm::None,
            length: LengthMode::Sentence,
        }
    }
}

/// Built-in style ids, always registered (the CLI cycles exactly these).
pub const BUILTIN_STYLES: [&str; 3] = ["default", "du", "sie"];

/// Built-in spec for one of [`BUILTIN_STYLES`] (unknown ids fall back to
/// [`StyleSpec::default_style`]).
pub fn builtin_spec(id: &str) -> StyleSpec {
    match id {
        "du" => StyleSpec {
            id: "du".to_string(),
            language: LanguageSpec::Auto,
            address_form: AddressForm::Du,
            length: LengthMode::Sentence,
        },
        "sie" => StyleSpec {
            id: "sie".to_string(),
            language: LanguageSpec::Auto,
            address_form: AddressForm::Sie,
            length: LengthMode::Sentence,
        },
        _ => StyleSpec::default_style(),
    }
}

/// Style resolved for a given field / cursor position: the winning spec id
/// plus its effective address form and length.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedStyle {
    /// Active style id (e.g. `"default"`).
    pub style_id: String,
    /// Effective address constraint.
    pub address_form: AddressForm,
    /// Effective length policy.
    pub length: LengthMode,
}

impl ResolvedStyle {
    /// The default style used when nothing else is configured.
    pub fn default_style() -> Self {
        Self {
            style_id: "default".to_string(),
            address_form: AddressForm::None,
            length: LengthMode::Sentence,
        }
    }

    /// Build a resolved style from an id (no constraints).
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            style_id: id.into(),
            address_form: AddressForm::None,
            length: LengthMode::Sentence,
        }
    }

    /// Resolve from a full spec, keeping its id.
    pub fn from_spec(spec: &StyleSpec) -> Self {
        Self {
            style_id: spec.id.clone(),
            address_form: spec.address_form,
            length: spec.length,
        }
    }
}

/// Du-family tokens (matched case-insensitively), forbidden in Sie mode.
pub const DU_BANNED: &[&str] = &[
    "du", "dich", "dir", "dein", "deine", "deinem", "deinen", "deiner", "deines",
];

/// Sie-family tokens (matched case-SENSITIVELY — lowercase `sie`/`ihr` are
/// ambiguous and never banned), forbidden in du mode.
pub const SIE_BANNED: &[&str] = &[
    "Sie", "Ihnen", "Ihr", "Ihre", "Ihrem", "Ihren", "Ihrer", "Ihres",
];

/// Banned words for an address form: the list plus whether matching is
/// case-sensitive (`None` has no bans).
pub fn banned_words(form: AddressForm) -> (&'static [&'static str], bool) {
    match form {
        AddressForm::Du => (SIE_BANNED, true),
        AddressForm::Sie => (DU_BANNED, false),
        AddressForm::None => (&[], false),
    }
}

/// Extra informal markers that only count for *detection* (never banned —
/// the ban lists follow the plan exactly). `euch`/`uns` are unambiguously
/// informal plural (formal would be `Ihnen`/`sich`).
const DU_DETECT_EXTRA: &[&str] = &["euch", "uns"];

/// Detect the address form from preceding text by simple token rules:
/// du-family (any case) vs capitalized Sie-family. Ties and blanks yield
/// `None`.
pub fn detect_address(text: &str) -> Option<AddressForm> {
    let (mut du, mut sie) = (0u32, 0u32);
    for raw in text.split(|c: char| !c.is_alphanumeric()) {
        if raw.is_empty() {
            continue;
        }
        let lower = raw.to_lowercase();
        if DU_BANNED.contains(&lower.as_str()) || DU_DETECT_EXTRA.contains(&lower.as_str()) {
            du += 1;
        }
        if SIE_BANNED.contains(&raw) {
            sie += 1;
        }
    }
    match du.cmp(&sie) {
        std::cmp::Ordering::Greater => Some(AddressForm::Du),
        std::cmp::Ordering::Less => Some(AddressForm::Sie),
        std::cmp::Ordering::Equal => None,
    }
}

/// True when `text` contains a token forbidden under `form`.
pub fn violates(text: &str, form: AddressForm) -> bool {
    let (banned, case_sensitive) = banned_words(form);
    if banned.is_empty() {
        return false;
    }
    text.split(|c: char| !c.is_alphanumeric()).any(|raw| {
        if raw.is_empty() {
            return false;
        }
        if case_sensitive {
            banned.contains(&raw)
        } else {
            banned.contains(&raw.to_lowercase().as_str())
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_specs_cover_ids() {
        assert_eq!(builtin_spec("default").address_form, AddressForm::None);
        assert_eq!(builtin_spec("du").address_form, AddressForm::Du);
        assert_eq!(builtin_spec("sie").address_form, AddressForm::Sie);
        assert_eq!(builtin_spec("nope").id, "default");
        assert_eq!(BUILTIN_STYLES.len(), 3);
    }

    #[test]
    fn detection_counts_unambiguous_markers() {
        assert_eq!(detect_address("kannst du mir helfen"), Some(AddressForm::Du));
        assert_eq!(
            detect_address("Können Sie mir helfen"),
            Some(AddressForm::Sie)
        );
        assert_eq!(detect_address("bitte prüfe deinen Bericht"), Some(AddressForm::Du));
        assert_eq!(detect_address(""), None);
        assert_eq!(detect_address("hello world"), None);
        // Lowercase sie/ihr decide nothing.
        assert_eq!(detect_address("sie geht nach hause"), None);
        assert_eq!(detect_address("the der"), None);
    }

    #[test]
    fn violations_follow_the_plan_lists() {
        // Sie mode forbids du-family (any case), du mode the capitalized form.
        assert!(violates("kannst du das prüfen", AddressForm::Sie));
        assert!(violates("Prüfe Dein Werk", AddressForm::Sie));
        assert!(violates("können Sie das prüfen", AddressForm::Du));
        // Ambiguous words are never banned.
        assert!(!violates("sie geht nach hause", AddressForm::Sie));
        assert!(!violates("sie geht nach hause", AddressForm::Du));
        assert!(!violates("gebt ihr das Buch", AddressForm::Du));
        assert!(!violates("anything at all", AddressForm::None));
        assert!(!violates("", AddressForm::Sie));
    }
}
