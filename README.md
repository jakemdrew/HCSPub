# HashChainStream & HCA Reference — Private / Not for Public Use

**NOTICE:** This repository is private and the code is not licensed for public use.
No permission is granted to copy, distribute, modify, or use any portion of this work.
All rights reserved © Jacob Michael Drew.

## Overview

This repository contains reference embodiments related to HashChainStream (HCS) and Hash-Chain Augmentation (HCA) data structures (e.g., radix/HAMT-style maps and anchor dictionaries).
The code implements prefix-stable, random-access, domain-separated hash streams and self-healing indexing policies.

## Status & Access

**License:** No open-source license is granted. All rights reserved.

**Use:** Internal evaluation only. Do not share binaries, source, docs, or artifacts.

**Redistribution:** Prohibited.

**Commercial use:** Prohibited without prior written consent.

**For inquiries (licensing, collaboration, research access):**
Jacob Michael Drew • jakemdrew@gmail.com 

## Patent Notice

**This code and associated designs are protected by the following patent family and filings:**

### Patent 1 (U.S. Provisional No. 63/845,399) — HashChain (deterministic, multi-seed, shard-aware addresses).

**Support:** HashChainStream (HCS) provides the underlying prefix-stable, extendable entropy needed to realize stable shard prefixes and composable directory layouts. Because earlier nibbles never change, shard paths remain valid as additional entropy is consumed for disambiguation, directly reinforcing deterministic addressing and shard awareness.

### Patent 2 (U.S. Provisional No. 63/847,656) — Stateless CAS with deterministic placement/deduplication using HashChain IDs.

**Support:** HCS enables a stateless, reconstructible addressing layer: any content key’s addressable coordinates (depth, prefix) are pure functions of (key, policy, domain). Containers (chunks, anchors) can be rebuilt from keys alone, eliminating the need for manifests or stateful routers while preserving deduplication semantics.

### Patent 3 (U.S. Provisional No. 63/849,997) — .nhash file format with structured footers (aliases, tags, provenance, compression).

**Support:** HCS provides a canonical, versioned namespace (via person tags) to bind semantic footers to deterministic addresses. This ensures footer metadata and derived views remain verifiably tied to stable prefixes, enabling in-situ introspection and consistent aliasing across evolutions.

### Patent 4 (U.S. Provisional No. 63/849,997) — Decentralized, self-healing indexing where indices are reproducible views from embedded footers.

**Support:** With HCS, index materialization becomes a deterministic function of the keys and footers: anchors/partitions are independently reconstructible without global scans, and damaged shards are repaired by recomputing only the affected (depth, prefix) regions. This realizes the family’s decentralized, resilient self-healing architecture.

### Patent 5 (U.S. Provisional No. 63/879,918) — Self-healing XBRL processing (SEC filings).

**Support:** HCS stabilizes feature addressing and taxonomy resolution by providing deterministic, shard-stable partitions over filings, references, and facts. When external taxonomies move or disappear, indices and caches are re-derivable from keys and footers, preserving model inputs and enabling localized repair without re-ingesting the corpus.

### Patent 6 (U.S. Provisional No. 63/882,206) — System and Method for Streaming Infinite Deterministic Hash Chains.

**Support:** The invention enables prefix stability, domain separation, and extendable entropy generation suitable for self-healing data structures, verifiable addressing, cryptographic primitives, and large-scale indexing.

## Usage Restrictions

No license granted. You may not reproduce, distribute, sublicense, or create derivative works.

No benchmarks or disclosures without written permission.

No third-party sharing (including screenshots, snippets, or API descriptions).

## Security & Confidentiality

By accessing this repository, you acknowledge the materials are confidential and proprietary.
Unauthorized use or disclosure may violate trade secret, copyright, and patent laws.

## Contact

**For permissions, licensing discussions, or research collaborations, contact:**
Jacob Michael Drew • jakemdrew@gmail.com
