//! IBus wire types: exact D-Bus encodings from the ibus source.
//! Every serializable starts with its type-name string, then the `a{sv}`
//! attachment map: `IBusText` is `('IBusText', a{sv}, text, attrs-variant)`,
//! attributes are `(tag, attachments, type, value, start, end)`, lookup
//! tables carry variant candidates and labels.
//!
//! The leading tag is load-bearing: the daemon dispatches on it, and a
//! missing tag fails closed with `bus_engine_proxy_g_signal` criticals.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use zbus::zvariant::{OwnedValue, Type, Value};

/// One text attribute: `(tag, attachments, type, value, start, end)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type, Value, OwnedValue)]
pub struct IBusAttribute {
    pub tag: String,
    pub attachments: HashMap<String, OwnedValue>,
    pub attr_type: u32,
    pub value: u32,
    pub start_index: u32,
    pub end_index: u32,
}

/// Underline attribute type id (IBUS_ATTR_TYPE_UNDERLINE).
pub const ATTR_TYPE_UNDERLINE: u32 = 1;
/// Single underline value (IBUS_ATTR_UNDERLINE_SINGLE).
pub const UNDERLINE_SINGLE: u32 = 1;

/// Attribute list: `(tag, attachments, variant-wrapped attributes)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type, Value, OwnedValue)]
pub struct IBusAttrList {
    pub tag: String,
    pub attachments: HashMap<String, OwnedValue>,
    pub attributes: Vec<OwnedValue>,
}

impl IBusAttrList {
    /// Empty attribute list.
    pub fn empty() -> Self {
        Self {
            tag: "IBusAttrList".to_string(),
            attachments: HashMap::new(),
            attributes: Vec::new(),
        }
    }

    /// Single underline over `[start, end)`.
    pub fn underline(start: u32, end: u32) -> Self {
        let attr = IBusAttribute {
            tag: "IBusAttribute".to_string(),
            attachments: HashMap::new(),
            attr_type: ATTR_TYPE_UNDERLINE,
            value: UNDERLINE_SINGLE,
            start_index: start,
            end_index: end,
        };
        Self {
            tag: "IBusAttrList".to_string(),
            attachments: HashMap::new(),
            attributes: vec![OwnedValue::try_from(attr).unwrap_or_else(|_| {
                IBusAttrList::empty().to_owned_value()
            })],
        }
    }

    fn to_owned_value(&self) -> OwnedValue {
        // Only used as an infallible fallback above; serialization of a
        // plain struct cannot realistically fail.
        OwnedValue::try_from(self.clone()).expect("attr list must serialize")
    }
}

/// IBus text: `(tag, attachments, UTF-8 string, attribute-list variant)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type, Value, OwnedValue)]
pub struct IBusText {
    pub tag: String,
    pub attachments: HashMap<String, OwnedValue>,
    pub text: String,
    pub attrs: OwnedValue,
}

impl IBusText {
    /// Plain text without attributes.
    pub fn plain(text: impl Into<String>) -> Self {
        Self {
            tag: "IBusText".to_string(),
            attachments: HashMap::new(),
            text: text.into(),
            attrs: OwnedValue::try_from(IBusAttrList::empty())
                .expect("attr list must serialize"),
        }
    }

    /// Text with the whole range underlined (preedit ghost).
    pub fn underlined(text: &str) -> Self {
        let len = text.chars().count() as u32;
        Self {
            tag: "IBusText".to_string(),
            attachments: HashMap::new(),
            text: text.to_string(),
            attrs: OwnedValue::try_from(IBusAttrList::underline(0, len))
                .expect("attr list must serialize"),
        }
    }

    /// Wrap for D-Bus signal arguments.
    pub fn into_variant(self) -> OwnedValue {
        OwnedValue::try_from(self).expect("text must serialize")
    }
}

/// IBus lookup table: `(tag, attachments, paging, variant candidates/labels)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type, Value, OwnedValue)]
pub struct IBusLookupTable {
    pub tag: String,
    pub attachments: HashMap<String, OwnedValue>,
    pub page_size: u32,
    pub cursor_pos: u32,
    pub cursor_visible: bool,
    pub round: bool,
    pub orientation: i32,
    pub candidates: Vec<OwnedValue>,
    pub labels: Vec<OwnedValue>,
}

impl IBusLookupTable {
    /// Horizontal table of plain-text candidates, cursor on the first.
    pub fn from_words(words: &[String]) -> Self {
        let candidates = words
            .iter()
            .map(|word| IBusText::plain(word).into_variant())
            .collect();
        Self {
            tag: "IBusLookupTable".to_string(),
            attachments: HashMap::new(),
            page_size: 5,
            cursor_pos: 0,
            cursor_visible: true,
            round: true,
            orientation: 0,
            candidates,
            labels: Vec::new(),
        }
    }

    /// Wrap for D-Bus signal arguments.
    pub fn into_variant(self) -> OwnedValue {
        OwnedValue::try_from(self).expect("table must serialize")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_signature_matches_ibus() {
        // (tag s, attachments a{sv}, text s, attrs variant)
        assert_eq!(IBusText::SIGNATURE.to_string(), "(sa{sv}sv)");
    }

    #[test]
    fn table_signature_matches_ibus() {
        // (tag, attachments, page u, cursor u, visible b, round b,
        //  orientation i, candidates av, labels av)
        assert_eq!(
            IBusLookupTable::SIGNATURE.to_string(),
            "(sa{sv}uubbiavav)"
        );
    }

    #[test]
    fn attr_signatures_match_ibus() {
        assert_eq!(IBusAttrList::SIGNATURE.to_string(), "(sa{sv}av)");
        assert_eq!(IBusAttribute::SIGNATURE.to_string(), "(sa{sv}uuuu)");
    }

    #[test]
    fn text_roundtrips_through_variant() {
        let text = IBusText::underlined("hello");
        let variant = text.clone().into_variant();
        let back: IBusText = variant.try_into().expect("decode");
        assert_eq!(back, text);
        assert_eq!(back.text, "hello");
    }

    #[test]
    fn table_roundtrips_through_variant() {
        let table = IBusLookupTable::from_words(
            &["one".to_string(), "two".to_string()],
        );
        assert_eq!(table.candidates.len(), 2);
        let variant = table.clone().into_variant();
        let back: IBusLookupTable = variant.try_into().expect("decode");
        assert_eq!(back, table);
    }

    #[test]
    fn underline_attr_marks_full_range() {
        let list = IBusAttrList::underline(0, 5);
        assert_eq!(list.attributes.len(), 1);
        let attr: IBusAttribute = list.attributes[0].clone().try_into().expect("decode");
        assert_eq!(
            (attr.attr_type, attr.value, attr.start_index, attr.end_index),
            (ATTR_TYPE_UNDERLINE, UNDERLINE_SINGLE, 0, 5)
        );
    }

    #[test]
    fn plain_has_empty_attrs() {
        let text = IBusText::plain("x");
        let list: IBusAttrList = text.attrs.try_into().expect("decode");
        assert!(list.attributes.is_empty());
    }
}
