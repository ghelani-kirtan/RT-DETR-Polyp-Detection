# Documentation Index

This document provides an overview of all documentation files in the project.

## 📋 Quick Reference

| Document | Purpose | When to Use |
|----------|---------|-------------|
| **QUICK_START.md** | Quick reference for common commands | Starting work, need command syntax |
| **RESTRUCTURE_SUMMARY.md** | High-level overview of changes | Understanding what changed |
| **PROJECT_RESTRUCTURE_GUIDE.md** | Detailed restructure documentation | Deep dive into changes |
| **README.md** | Project overview | First-time setup, project info |

## 🔧 Verification & Testing

| Script | Purpose | When to Use |
|--------|---------|-------------|
| **verify_structure.py** | Verify project structure | After restructure, before commit |
| **test_workflows.py** | Test critical workflows | Ensure functionality intact |

## 📚 Documentation Files

### Core Documentation

#### QUICK_START.md
- **Purpose**: Quick reference for common commands
- **Contains**:
  - Installation instructions
  - Common command examples
  - Project structure overview
  - Environment setup
- **Use When**: You need to quickly find a command or setup instruction

#### RESTRUCTURE_SUMMARY.md
- **Purpose**: Executive summary of restructure
- **Contains**:
  - Statistics and overview
  - Before/after comparison
  - Key improvements
  - Verification results
- **Use When**: You want a high-level understanding of what changed

#### PROJECT_RESTRUCTURE_GUIDE.md
- **Purpose**: Comprehensive restructure documentation
- **Contains**:
  - Detailed directory structure
  - Complete list of changes
  - Migration guide
  - How to run all components
  - Installation instructions
- **Use When**: You need detailed information about the restructure

#### README.md
- **Purpose**: Main project documentation
- **Contains**:
  - Project overview
  - Original setup instructions
  - General information
- **Use When**: First time working with the project

### Verification Scripts

#### verify_structure.py
- **Purpose**: Automated structure verification
- **Checks**:
  - All new files exist
  - All old files removed
  - Python syntax validity
- **Usage**: `python verify_structure.py`
- **Use When**: After restructure or before committing changes

#### test_workflows.py
- **Purpose**: Test critical workflows
- **Tests**:
  - Import paths
  - Module structure
  - File locations
  - Core functionality
- **Usage**: `python test_workflows.py`
- **Use When**: Ensuring functionality is intact

### Module-Specific Documentation

#### scripts/data/README.md
- **Purpose**: Data pipeline documentation
- **Contains**: Data pipeline usage and configuration
- **Use When**: Working with data pipelines

#### scripts/data/MIGRATION_GUIDE.md
- **Purpose**: Data pipeline migration guide
- **Contains**: Migration instructions for data pipelines
- **Use When**: Migrating data pipeline code

#### scripts/data/ROOT_FIXES_VS_PATCHES.md
- **Purpose**: Documentation of fixes vs patches
- **Contains**: Explanation of root fixes approach
- **Use When**: Understanding fix methodology

#### scripts/inference/README.md
- **Purpose**: Inference documentation
- **Contains**: Inference script usage
- **Use When**: Running inference

#### scripts/training_commands.txt
- **Purpose**: Training command examples
- **Contains**: Various training command examples
- **Use When**: Training models

#### tools/README.md
- **Purpose**: Tools documentation
- **Contains**: Training and export tool usage
- **Use When**: Using training/export tools

## 🗂️ Documentation by Use Case

### Getting Started
1. Start with **README.md** for project overview
2. Read **QUICK_START.md** for setup and common commands
3. Run **verify_structure.py** to ensure correct setup

### Understanding the Restructure
1. Read **RESTRUCTURE_SUMMARY.md** for overview
2. Refer to **PROJECT_RESTRUCTURE_GUIDE.md** for details
3. Run **test_workflows.py** to verify functionality

### Training Models
1. Check **scripts/training_commands.txt** for examples
2. Refer to **tools/README.md** for tool documentation
3. Use **QUICK_START.md** for quick command reference

### Running Inference
1. Check **scripts/inference/README.md** for inference docs
2. Use **QUICK_START.md** for command examples
3. Refer to **PROJECT_RESTRUCTURE_GUIDE.md** for file locations

### Working with Data
1. Read **scripts/data/README.md** for pipeline docs
2. Check **scripts/data/MIGRATION_GUIDE.md** if migrating
3. Use **QUICK_START.md** for command examples

### Cloud Operations
1. Use **QUICK_START.md** for S3 sync commands
2. Refer to **PROJECT_RESTRUCTURE_GUIDE.md** for details
3. Check **requirements/cloud-requirements.txt** for dependencies

## 📦 Requirements Documentation

| File | Purpose |
|------|---------|
| **requirements/requirements.txt** | Base dependencies |
| **requirements/desktop-requirements.txt** | Desktop/GUI dependencies |
| **requirements/mac-requirements.txt** | Mac-specific dependencies |
| **requirements/inference-requirements.txt** | Inference dependencies |
| **requirements/cloud-requirements.txt** | Cloud/S3 dependencies |

## 🔍 Finding Information

### "How do I train a model?"
→ **QUICK_START.md** or **scripts/training_commands.txt**

### "What changed in the restructure?"
→ **RESTRUCTURE_SUMMARY.md** or **PROJECT_RESTRUCTURE_GUIDE.md**

### "Where is file X now?"
→ **PROJECT_RESTRUCTURE_GUIDE.md** (see "Files Moved" section)

### "How do I run inference?"
→ **QUICK_START.md** or **scripts/inference/README.md**

### "How do I prepare data?"
→ **scripts/data/README.md** or **QUICK_START.md**

### "How do I sync with S3?"
→ **QUICK_START.md** (Cloud Sync section)

### "Is the restructure correct?"
→ Run **verify_structure.py** and **test_workflows.py**

## 📝 Documentation Maintenance

When updating the project:
1. Update relevant documentation files
2. Run verification scripts
3. Update this index if adding new docs
4. Keep QUICK_START.md current with common commands

## 🎯 Recommended Reading Order

### For New Users
1. README.md
2. QUICK_START.md
3. Relevant module documentation

### For Understanding Restructure
1. RESTRUCTURE_SUMMARY.md
2. PROJECT_RESTRUCTURE_GUIDE.md
3. Run verification scripts

### For Development
1. QUICK_START.md (keep handy)
2. Module-specific README files
3. Training commands reference

---

**Last Updated**: After project restructure
**Maintained By**: Project team
