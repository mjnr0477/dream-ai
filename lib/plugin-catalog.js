const CATEGORIES = Object.freeze([
  { id: "research", name: "Research and knowledge" },
  { id: "social", name: "Social networks and communities" },
  { id: "communication", name: "Messaging and communication" },
  { id: "video_audio", name: "Video, music and live media" },
  { id: "shopping", name: "Shopping and marketplaces" },
  { id: "productivity", name: "Productivity and cloud storage" },
  { id: "developer", name: "Developer platforms" },
  { id: "finance", name: "Finance and payments" },
  { id: "travel", name: "Travel and maps" },
  { id: "education", name: "Education and learning" }
]);

// This is a capability catalogue, not an assertion that every listed service is
// connected. "connected_read_only" means a public retrieval connector exists in
// this codebase; other entries are explicitly unavailable until approved setup.
const PLUGINS = Object.freeze([
  { id: "wikipedia", name: "Wikipedia", category: "research", status: "connected_read_only", access: "anonymous_public_read", apiKeyFree: true, capabilities: ["search_articles", "retrieve_short_excerpts"], note: "Article text has its own license and attribution requirements." },
  { id: "wikidata", name: "Wikidata", category: "research", status: "connected_read_only", access: "anonymous_public_read", apiKeyFree: true, capabilities: ["search_public_entities", "retrieve_structured_facts"], note: "Preserve provenance and screen personal or sensitive records." },
  { id: "gdelt", name: "GDELT news discovery", category: "research", status: "connected_read_only", access: "anonymous_public_read", apiKeyFree: true, capabilities: ["discover_news_headlines", "retrieve_publisher_links"], note: "A headline is a lead, not proof of an event or a license to train on publisher articles." },
  { id: "google_news_rss", name: "Google News RSS", category: "research", status: "connected_read_only", access: "anonymous_public_read", apiKeyFree: true, capabilities: ["discover_news_headlines", "retrieve_publication_dates"], note: "RSS access does not grant rights to reuse linked publisher articles for training." },
  { id: "crossref", name: "Crossref", category: "research", status: "connected_read_only", access: "anonymous_public_read", apiKeyFree: true, capabilities: ["search_scholarly_metadata", "retrieve_doi_metadata"], note: "Metadata is distinct from abstracts and full-text publication rights." },
  { id: "openalex", name: "OpenAlex", category: "research", status: "connected_read_only", access: "anonymous_public_read", apiKeyFree: true, capabilities: ["search_scholarly_works", "retrieve_citation_metadata"], note: "Anonymous access has lower quotas than authenticated access." },
  { id: "europe_pmc", name: "Europe PMC", category: "research", status: "connected_read_only", access: "anonymous_public_read", apiKeyFree: true, capabilities: ["search_biomedical_literature", "retrieve_abstracts_and_open_access_links"], note: "Check the license of each article before reuse or training." },
  { id: "stack_exchange", name: "Stack Exchange", category: "research", status: "connected_read_only", access: "anonymous_public_read", apiKeyFree: true, capabilities: ["search_public_technical_questions", "retrieve_attributed_excerpts"], note: "Respect shared-IP quotas, attribution and share-alike terms." },
  { id: "bluesky", name: "Bluesky public posts", category: "social", status: "connected_read_only", access: "anonymous_public_read", apiKeyFree: true, capabilities: ["search_public_posts"], note: "Public-post retrieval only; it does not access private feeds or authorize model training." },
  { id: "public_github", name: "GitHub public repositories", category: "developer", status: "catalogued", access: "anonymous_public_read_rate_limited", apiKeyFree: true, capabilities: ["planned_repository_discovery", "planned_release_and_issue_lookup"], note: "Not yet connected in Dream AI's runtime. Public code must be license-checked before reuse or training." },
  { id: "huggingface_datasets", name: "Hugging Face public datasets", category: "research", status: "catalogued", access: "anonymous_public_read_for_non_gated_datasets", apiKeyFree: true, capabilities: ["planned_dataset_discovery", "planned_metadata_and_sample_lookup"], note: "Private and gated datasets are excluded; dataset cards and underlying data rights need review." },
  { id: "pubmed", name: "NCBI PubMed", category: "research", status: "catalogued", access: "anonymous_public_read_rate_limited", apiKeyFree: true, capabilities: ["planned_bibliographic_search", "planned_abstract_metadata"], note: "Ordinary use can be keyless; linked article rights remain separate." },
  { id: "world_bank", name: "World Bank open data", category: "research", status: "catalogued", access: "anonymous_public_read", apiKeyFree: true, capabilities: ["planned_country_indicators", "planned_development_statistics"], note: "Check individual dataset licenses and third-party exceptions." },
  { id: "ckan_open_data", name: "Public CKAN data portals", category: "research", status: "catalogued", access: "anonymous_public_read_where_portal_allows", apiKeyFree: true, capabilities: ["planned_open_dataset_discovery"], note: "CKAN itself is not a blanket data license; check each dataset and resource." },
  { id: "common_crawl", name: "Common Crawl", category: "research", status: "catalogued", access: "public_archive_discovery", apiKeyFree: true, capabilities: ["planned_public_web_archive_discovery"], note: "Archive access is not a blanket license to retain or train on crawled pages." },
  { id: "open_library", name: "Open Library", category: "education", status: "catalogued", access: "public_read_subject_to_usage_guidelines", apiKeyFree: true, capabilities: ["planned_book_and_edition_metadata"], note: "Use conservatively for catalogue discovery and follow its application guidance." },
  { id: "x", name: "X public posts", category: "social", status: "requires_provider_access", access: "official_api_and_provider_credentials", apiKeyFree: false, capabilities: ["planned_public_post_search"], note: "No X token is configured. Dream AI must not scrape around access controls." },
  { id: "facebook_instagram_threads", name: "Facebook, Instagram and Threads", category: "social", status: "requires_provider_access", access: "approved_meta_api_and_user_authorization", apiKeyFree: false, capabilities: ["planned_approved_public_content", "planned_user_authorized_account_actions"], note: "No unrestricted anonymous access; app review and permission scopes may be required." },
  { id: "tiktok", name: "TikTok", category: "social", status: "requires_provider_access", access: "approved_tiktok_api_and_scoped_authorization", apiKeyFree: false, capabilities: ["planned_approved_content_discovery"], note: "Only approved API products and permission scopes may be used." },
  { id: "reddit", name: "Reddit", category: "social", status: "requires_provider_access", access: "official_api_and_terms_review", apiKeyFree: false, capabilities: ["planned_permitted_public_content_discovery"], note: "Commercial use and AI training require separate terms and rights review." },
  { id: "linkedin", name: "LinkedIn", category: "social", status: "requires_provider_access", access: "approved_api_and_user_authorization", apiKeyFree: false, capabilities: ["planned_user_authorized_profile_and_page_actions"], note: "Access depends on the approved LinkedIn API product and scopes." },
  { id: "telegram", name: "Telegram", category: "communication", status: "requires_user_authorization", access: "bot_setup_or_user_authorized_connection", apiKeyFree: false, capabilities: ["planned_bot_messaging", "planned_channel_tools"], note: "A bot token or authorized connection is required; Dream AI will not ask users to paste secrets into chat." },
  { id: "discord", name: "Discord", category: "communication", status: "requires_user_authorization", access: "approved_bot_installation_and_scopes", apiKeyFree: false, capabilities: ["planned_server_bot_tools", "planned_authorized_messaging"], note: "A server administrator must approve bot installation and permissions." },
  { id: "whatsapp", name: "WhatsApp", category: "communication", status: "requires_provider_access", access: "official_business_platform_and_authorization", apiKeyFree: false, capabilities: ["planned_business_messaging"], note: "No private-message scraping or access-control bypass." },
  { id: "gmail", name: "Gmail", category: "communication", status: "requires_user_authorization", access: "oauth_user_consent", apiKeyFree: false, capabilities: ["planned_search_and_draft", "planned_send_with_confirmation"], note: "Each account owner must authorize the permitted scopes." },
  { id: "youtube", name: "YouTube", category: "video_audio", status: "requires_provider_access", access: "youtube_data_api_key_or_user_authorization", apiKeyFree: false, capabilities: ["planned_video_metadata_search", "planned_authorized_channel_actions"], note: "The official Data API is not currently configured; viewing or metadata discovery is not the same as downloading videos." },
  { id: "spotify", name: "Spotify", category: "video_audio", status: "requires_user_authorization", access: "spotify_oauth_and_api_access", apiKeyFree: false, capabilities: ["planned_music_discovery", "planned_authorized_library_actions"], note: "Playback and account actions remain subject to Spotify's API rules." },
  { id: "twitch", name: "Twitch", category: "video_audio", status: "requires_provider_access", access: "official_twitch_api_and_scoped_authorization", apiKeyFree: false, capabilities: ["planned_live_stream_discovery", "planned_authorized_channel_tools"], note: "Use official endpoints and respect broadcaster and platform rights." },
  { id: "soundcloud", name: "SoundCloud", category: "video_audio", status: "requires_provider_access", access: "approved_soundcloud_api_access", apiKeyFree: false, capabilities: ["planned_track_and_artist_discovery"], note: "Availability depends on current API access and usage terms." },
  { id: "netflix", name: "Netflix", category: "video_audio", status: "planned", access: "no_general_public_consumer_api", apiKeyFree: false, capabilities: ["planned_public_title_information_only"], note: "No general-purpose account or playback integration is assumed." },
  { id: "amazon", name: "Amazon", category: "shopping", status: "requires_provider_access", access: "eligible_partner_program_or_approved_api", apiKeyFree: false, capabilities: ["planned_product_catalogue_lookup"], note: "Partner approval and program terms may apply; this is not unrestricted marketplace access." },
  { id: "ebay", name: "eBay", category: "shopping", status: "requires_provider_access", access: "official_api_application_and_scopes", apiKeyFree: false, capabilities: ["planned_product_search", "planned_order_tools_with_authorization"], note: "Account actions require explicit authorization and confirmation." },
  { id: "jumia", name: "Jumia", category: "shopping", status: "planned", access: "partner_or_public_endpoint_if_offered", apiKeyFree: false, capabilities: ["planned_product_discovery"], note: "No supported anonymous marketplace connector is currently configured." },
  { id: "alibaba", name: "Alibaba", category: "shopping", status: "requires_provider_access", access: "official_partner_api_and_permissions", apiKeyFree: false, capabilities: ["planned_product_and_supplier_discovery"], note: "Use official partner services and their access terms." },
  { id: "shopify", name: "Shopify", category: "shopping", status: "requires_user_authorization", access: "store_owner_app_installation_and_scopes", apiKeyFree: false, capabilities: ["planned_authorized_store_management"], note: "A store owner must approve the app and requested permissions." },
  { id: "google_drive_calendar", name: "Google Drive and Calendar", category: "productivity", status: "requires_user_authorization", access: "google_oauth_and_scoped_consent", apiKeyFree: false, capabilities: ["planned_file_search", "planned_calendar_read_and_edit"], note: "Users authorize only the requested scopes; external actions require confirmation." },
  { id: "microsoft_365", name: "Microsoft 365", category: "productivity", status: "requires_user_authorization", access: "microsoft_oauth_and_scoped_consent", apiKeyFree: false, capabilities: ["planned_mail_file_and_calendar_tools"], note: "Users and, where required, organization administrators approve access." },
  { id: "notion", name: "Notion", category: "productivity", status: "requires_user_authorization", access: "notion_integration_authorization", apiKeyFree: false, capabilities: ["planned_authorized_workspace_search_and_edit"], note: "Workspace owners choose which pages and databases an integration can access." },
  { id: "dropbox", name: "Dropbox", category: "productivity", status: "requires_user_authorization", access: "oauth_user_consent", apiKeyFree: false, capabilities: ["planned_authorized_file_search_and_management"], note: "Access is limited to the scopes and files the account owner authorizes." },
  { id: "paypal", name: "PayPal", category: "finance", status: "requires_user_authorization", access: "merchant_or_account_authorization", apiKeyFree: false, capabilities: ["planned_payment_status_and_approved_transactions"], note: "Financial actions need explicit confirmation and secure authorization." },
  { id: "stripe", name: "Stripe", category: "finance", status: "requires_user_authorization", access: "account_owner_oauth_or_restricted_api_credentials", apiKeyFree: false, capabilities: ["planned_authorized_payment_and_billing_tools"], note: "Never expose payment secrets to the browser or ask users to paste them into chat." },
  { id: "mpesa", name: "M-PESA", category: "finance", status: "requires_provider_access", access: "safaricom_business_api_and_owner_authorization", apiKeyFree: false, capabilities: ["planned_authorized_payment_and_status_tools"], note: "Use official Safaricom integrations; payment actions require explicit confirmation." },
  { id: "openstreetmap", name: "OpenStreetMap", category: "travel", status: "planned", access: "public_services_subject_to_usage_policy", apiKeyFree: true, capabilities: ["planned_map_and_geocoding_lookup"], note: "Respect public service usage policies; do not build unrestricted high-volume scraping." },
  { id: "booking_skyscanner", name: "Booking.com and Skyscanner", category: "travel", status: "requires_provider_access", access: "approved_partner_api_or_affiliate_access", apiKeyFree: false, capabilities: ["planned_travel_search"], note: "Availability and prices must come from authorized current data sources." },
  { id: "uber", name: "Uber", category: "travel", status: "requires_provider_access", access: "approved_uber_api_and_user_authorization", apiKeyFree: false, capabilities: ["planned_authorized_ride_estimates_and_requests"], note: "Ride requests require user authorization and final confirmation." },
  { id: "khan_academy", name: "Khan Academy", category: "education", status: "planned", access: "public_content_and_official_services_subject_to_terms", apiKeyFree: true, capabilities: ["planned_learning_resource_discovery"], note: "Public learning resources can be discovered; this does not imply unrestricted content reuse." },
  { id: "coursera", name: "Coursera", category: "education", status: "planned", access: "public_catalogue_or_approved_integration", apiKeyFree: false, capabilities: ["planned_course_discovery"], note: "Course access and content rights remain with the provider." }
]);

export function getDreamPluginCatalog() {
  const plugins = PLUGINS.map(plugin => ({
    ...plugin,
    capabilities: [...plugin.capabilities]
  }));
  return {
    version: 1,
    mode: "discovery_catalog",
    guestAccessSupported: true,
    userRegistrationRequired: false,
    userSuppliedApiKeysRequired: false,
    policy: {
      anonymousPublicReadOnlyByDefault: true,
      privateAccountAccessRequiresAuthorization: true,
      writeAndFinancialActionsRequireExplicitConfirmation: true,
      publicAccessDoesNotGrantTrainingRights: true,
      unconfiguredConnectorsAreNeverReportedAsConnected: true
    },
    categories: CATEGORIES.map(category => ({ ...category })),
    plugins
  };
}
