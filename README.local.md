# README
HOME直下に2つのrepositoryを配置する想定です

```
~/ucllm_nedo_prod
~/Megatron-DeepSpeed
```

```
git clone https://github.com/geniac-team-sannai/ucllm_nedo_prod.git
git checkout dev
```

## jsonlの作成
サンプルとしてwikipedia_enの一部を取得します

`~/ucllm_nedo_prod/train/scripts/step2_pretrain_model/gcp_node-1_gpu/wiki-en_gpt_tiny`以下にtrain.jsonlが出力されます

```
cd ~/ucllm_nedo_prod/train/scripts/step2_pretrain_model/gcp_node-1_gpu/wiki-en_gpt_tiny
python download_hf_ds.py
```

※ ここではphi-2のtokenizerを使う

`~/ucllm_nedo_prod/train/scripts/step2_pretrain_model/gcp_node-1_gpu/wiki-en_gpt_tiny`以下の`prepare_dataset.sh`の
`megatron_deepspeed_dir="/home/ext_otomijuf_004_gmail_com/Megatron-DeepSpeed"`は適宜修正してください

```
cd ~/ucllm_nedo_prod/train/scripts/step2_pretrain_model/gcp_node-1_gpu/wiki-en_gpt_tiny
sh prepare_dataset.sh
```

※ megatron-coreがライブラリとして入っている場合エラーが起きる?  
原因は、`prepare_dataset.sh`が呼び出している`Megatron-DeepSpeed/tools/preprocess_data.py`がmegatron-coreのbuild_tokenizerを呼び出してしまうため。`Megatron-DeepSpeed/tools/preprocess_data.py`では、`Megatron-DeepSpeed/megatron.tokenizer.build_tokenizer`を呼び出せるよう対応する必要がある。  
とりあえずの対応として、`Megatron-DeepSpeed/tools/preprocess_data.py`の上の方に、`sys.path = [/(HOME)/Megatron-DeepSpeed/megatron] + sys.path`と加えれば動くはず  


## 事前学習
wandbにログイン
> # W&Bにログイン。
> # https://wandb.ai/settings --> Danger Zone --> API keys --> APIキーをコピペ。
> (.venv) $ wandb login

`~/ucllm_nedo_prod/train/scripts/step2_pretrain_model/gcp_node-1_gpu/wiki-en_gpt_tiny`にある`ds_config_gpt_TEMPLATE.json`のproject名を適宜修正

```
 "wandb": {
    "enabled": true,
    "project": "sample"
  }
```

※ たぶんここに書けばwandbに連携されるはず

事前学習の実行

```
bash zero-0_dp-1_pp-1_tp-1_flashattn2-on.sh \
--input_tokenizer_file "microsoft/phi-2" \
--output_model_dir "./output"
```

`./output`以下にlogやtensorboardやcheckpointが出力されます

#### 変更点1
以下のようなエラーが出ることがある。
```
torch.distributed.DistNetworkError: The server socket has failed to listen on any local network address. The server socket has failed to bind to [::]:29500 (errno: 98 - Address already in use). The server socket has failed to bind to ?UNKNOWN? (errno: 98 - Address already in use).
```

おそらく他の人が事前学習を動かしているとportが被る？？？

`~/ucllm_nedo_prod/train/scripts/step2_pretrain_model/gcp_node-1_gpu/wiki-en_gpt_tiny/zero-0_dp-1_pp-1_tp-1_flashattn2-on.sh`では、以下のようにportを変更するようにしてます。

```
DISTRIBUTED_ARGS="
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
```

※ マルチノードのときに使うやつ？変更の影響は不明

#### 変更点2
huggingfaceにあるphi-2のtokenizerを使えるように、`--tokenizer-type HFTokenizer`としてます。

```
data_options=" \
    --tokenizer-type HFTokenizer \
    --tokenizer-model ${input_tokenizer_file} \
    --data-path ${data_path} \
    --data-impl mmap"
```

※ 事前学習のどこでtokenizerを使ってるか調査


#### 変更点3
モデルサイズの変更

```
## GPT-3 TinyTiny (10M?)
model_size=0.01
num_layers=6
hidden_size=256
num_attn_heads=4
global_batch_size=128
lr=6.0e-4
min_lr=1.0e-6
init_std=0.02
```

※ 学習可能なパラメタ数を確認する必要がある。すでに標準出力されているのを見落としてるかも

#### 変更点4
deepspeedのconfigのtemplateファイルを新たに作成してます。

元のpipelineのコードでは、megatron_deepspeedにあるものを使ってるようです。
`template_json="${megatron_deepspeed_dir}/examples_deepspeed/rebase/ds_config_gpt_TEMPLATE.json"`


今回作成したものには以下が追加されています。

```
 "wandb": {
    "enabled": true,
    "project": "sample"
  }
```

※ wandbにデータが飛んでるのを確認できてない
