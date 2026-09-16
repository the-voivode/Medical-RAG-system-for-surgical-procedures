# # create_user.py
# import argparse
# from auth import AuthManager

# p = argparse.ArgumentParser()
# p.add_argument("--username", required=True)
# p.add_argument("--password", required=True)
# p.add_argument("--api-key",  required=True, help="This user's own OpenRouter key (sk-or-...)")
# p.add_argument("--model", default="meta-llama/llama-3.1-8b-instruct")
# a = p.parse_args()

# AuthManager().add_user(a.username, a.password, a.api_key, a.model)
# print(f"✅ User '{a.username}' created (model: {a.model})")

# create_user.py
import argparse
from auth import AuthManager
from keyvault import KeyVault


def main():
    parser = argparse.ArgumentParser(
        description="Create a user and assign a backend OpenRouter API key."
    )

    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument(
        "--api-key",
        required=True,
        help="OpenRouter API key assigned to this user. Stored only in KeyVault."
    )
    parser.add_argument(
        "--key-alias",
        default=None,
        help="Optional alias for the API key. Defaults to username."
    )
    parser.add_argument(
        "--model",
        default="meta-llama/llama-3.1-8b-instruct"
    )

    args = parser.parse_args()

    auth = AuthManager()
    vault = KeyVault()

    alias = (args.key_alias or args.username).strip()

    auth.add_user(
        username=args.username,
        password=args.password,
        key_alias=alias,
        default_model=args.model
    )

    vault.set_key(alias, args.api_key)

    print(f"✅ User '{args.username}' created.")
    print(f"🔑 API key stored under alias '{alias}'.")


if __name__ == "__main__":
    main()