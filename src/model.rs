impl<A:AutodiffBackend<InnerBackend=B>,B:Backend,V:AutodiffModule<A,InnerModule=W>+BlockVariant<A>,W:BlockVariant<B>> AutodiffModule<A> for Model<A,V>{
	fn from_inner(inner:Self::InnerModule)->Self{
		Model{
			blocks:AutodiffModule::from_inner(inner.blocks),
			phantom:PhantomData,
			promptencoding:inner.promptencoding,
			prompt:inner.prompt,
			tokenizer:inner.tokenizer
		}
	}
	fn valid(&self)->Self::InnerModule{
		Model{
			blocks:self.blocks.valid(),
			phantom:PhantomData,
			promptencoding:self.promptencoding,
			prompt:self.prompt.clone(),
			tokenizer:self.tokenizer.clone()
		}
	}
	type InnerModule=Model<B,W>;
}
impl<B:Backend,V:BlockVariant<B>+ModuleDisplay> ModuleDisplay for Model<B,V>{}
impl<B:Backend,V:BlockVariant<B>+ModuleDisplay> ModuleDisplayDefault for Model<B,V>{
	fn content(&self,content:Content)->Option<Content>{self.blocks.content(content)}
}
impl<B:Backend,V:BlockVariant<B>> BlockVariant<B> for Model<B,V>{
	fn clear(&mut self){
		self.blocks.clear();
		self.prompt.clear();
	}
	fn detach_cache(&mut self){self.blocks.detach_cache()}
	fn embed(&self,input:Tensor<B,2,Int>,inputclasses:usize,inputencoding:u64)->Value<B>{
		if self.prompt.len()==0{return self.blocks.embed(input,inputclasses,inputencoding)}
		self.clone().embed_mut(input,inputclasses,inputencoding)
	}
	fn embed_mut(&mut self,input:Tensor<B,2,Int>,inputclasses:usize,inputencoding:u64)->Value<B>{
		self.do_prompt();
		self.blocks.embed_mut(input,inputclasses,inputencoding)
	}
	fn encoding_hint(&self)->Option<u64>{Some(self.promptencoding.clone())}
	fn forward(&self,input:Value<B>)->Value<B>{
		if self.prompt.len()==0{return self.blocks.forward(input)}
		self.clone().forward_mut(input)
	}
	fn forward_mut(&mut self,input:Value<B>)->Value<B>{self.infer(input)}
	fn supports(&self,encoding:u64)->bool{self.blocks.supports(encoding)}
	fn tokenizer_hint(&self)->Option<TokenDict>{Some(self.tokenizer.clone())}
	type BlockWith<C:Backend>=Model<C,V::BlockWith<C>>;
}
impl<B:Backend,V:BlockVariant<B>> Model<B,V>{
	/// appends text to the prompt
	pub fn absorb_text(&mut self,text:&str){self.prompt.extend(self.tokenizer.tokenize_str(text))}
	/// reference the layers
	pub fn blocks(&self)->&Vec<V>{&self.blocks.0}
	/// reference the layers
	pub fn blocks_mut(&mut self)->&mut Vec<V>{&mut self.blocks.0}
	/// reads the cached prompt into the model. returns None no prompt is stored
	pub fn do_prompt(&mut self)->Option<Value<B>>{
		if self.prompt.len()==0{return None}

		let prompt=mem::take(&mut self.prompt);
		let seq=prompt.len();

		let input=Tensor::from_data(TensorData::new(prompt,[1,seq]),&Default::default());
		let inputclasses=self.tokenizer.len();
		let inputencoding=self.promptencoding;

		Some(self.blocks.embed_mut(input,inputclasses,inputencoding))
	}
	pub fn get_tokenizer(&self)->TokenDict{self.tokenizer.clone()}
	/// apply inference to the input value
	pub fn infer(&mut self,input:Value<B>)->Value<B>{
		self.do_prompt();
		self.blocks.forward_mut(input)
	}
	/// convert from a sequential block. note that while putting a block enum that is Into<Sequential<V>> in here is allowed, the variant type may remain wrapped even if that block enum wraps a Sequential<V>
	pub fn from_block<K:Into<Sequential<V>>>(block:K,encoding:u64,tokenizer:TokenDict)->Self{
		Self{
			blocks:block.into(),
			phantom:PhantomData,
			promptencoding:encoding,
			prompt:Vec::new(),
			tokenizer
		}
	}
	/// convert into a sequential block. the return type is generic to allow either getting a block enum or a raw Sequential<V>
	pub fn into_block<K:From<RecursiveVariant<Sequential<V>>>>(self)->K{RecursiveVariant::from(self.blocks).into()}
	/// wraps in a structure for training on token sequences
	pub fn into_token_seq_train(self)->TokenSeqTrain<Autodiff<B>,Model<Autodiff<B>,V::BlockWith<Autodiff<B>>>>{TokenSeqTrain::new(self)}
	/// loads from a file
	pub fn load<P:AsRef<Path>>(path:P)->IOResult<Self>{
		let path=path.as_ref();

		let file=File::open(path)?;
		let reader=BufReader::new(file);

		match rmp_decode::from_read(reader){Err(e)=>Err(IOError::new(IOErrorKind::Other,e.to_string())),Ok(x)=>Ok(x)}
	}
	/// predict the next token and reappend it to the prompt for autoregressive inference
	pub fn next(&mut self,input:&str)->Token{
		self.absorb_text(input);

		let id:u32=self.predict(None).into_scalar().elem();
		let token=self.tokenizer[id as usize].clone();

		self.prompt.push(id);
		token
	}
	/// predict the next tokens
	pub fn predict(&mut self,input:Option<Tensor<B,2,Int>>)->Tensor<B,1,Int>{
		let prioroutput=self.do_prompt();
		let output:Tensor<B,3>=if let Some(input)=input{
			let inputclasses=self.tokenizer.len();
			let inputencoding=self.promptencoding;

			self.blocks.embed_mut(input,inputclasses,inputencoding)
		}else{
			if let Some(p)=prioroutput{p}else{return Tensor::ones([1],&Default::default())}
		}.get_data();

		block::soft_choose(output.slice(s![..,-1,..]),-1,1.0).squeeze_dims(&[1,2])
	}
	/// sample
	pub fn sample<'a>(&'a mut self,input:&str)->impl 'a+Iterator<Item=char>{
		self.absorb_text(input);
		UTF8CharIter::from((0..).flat_map(move|_|self.next(""))).map(|x|x.unwrap_or(char::REPLACEMENT_CHARACTER))
	}
	/// saves to a file
	pub fn save<P:AsRef<Path>>(&self,path:P)->IOResult<()>{
		let file=File::create(path)?;
		let mut writer=BufWriter::new(file);

		match rmp_encode::write(&mut writer,self){Err(e)=>Err(IOError::new(IOErrorKind::Other,e.to_string())),Ok(x)=>Ok(x)}
	}
	/// set the prompt encoding id
	pub fn set_prompt_encoding(&mut self,encoding:u64){
		self.promptencoding=encoding;
	}
}
impl<B:Backend,V:BlockVariant<B>> Module<B> for Model<B,V>{
	fn collect_devices(&self,devices:Vec<B::Device>)->Vec<B::Device>{self.blocks.collect_devices(devices)}
	fn fork(mut self,device:&B::Device)->Self{
		self.blocks=self.blocks.fork(device);
		self
	}
	fn into_record(self)->Self::Record{self.blocks.into_record()}
	fn load_record(mut self,record:Self::Record)->Self{
		self.blocks=self.blocks.load_record(record);
		self
	}
	fn map<M:ModuleMapper<B>>(mut self,mapper:&mut M)->Self{
		self.blocks=self.blocks.map(mapper);
		self
	}
	fn to_device(mut self,device:&B::Device)->Self{
		self.blocks=self.blocks.to_device(device);
		self
	}
	fn visit<M:ModuleVisitor<B>>(&self,visitor:&mut M){self.blocks.visit(visitor)}
	type Record=<Sequential<V> as Module<B>>::Record;
}
#[derive(Clone,Debug,Default,Deserialize,Serialize)]
#[serde(bound="")]
/// model structure vaguely oriented towards language but whose exact abilities depend on what blocks are inside
pub struct Model<B:Backend,V:BlockVariant<B>>{blocks:Sequential<V>,phantom:PhantomData<B>,promptencoding:u64,prompt:Vec<u32>,tokenizer:TokenDict}

pub type BasicModel<B>=Model<B,Block<B>>;

use burn::{
	backend::Autodiff,
	module::{AutodiffModule,Content,ModuleDisplay,ModuleDisplayDefault,ModuleMapper,ModuleVisitor},
	prelude::*,
	tensor::backend::AutodiffBackend
};
use crate::{
	block::{
		Block,BlockVariant,RecursiveVariant,Value,multi::Sequential,self
	},
	train::TokenSeqTrain
};
use rmp_serde::{decode as rmp_decode,encode as rmp_encode};
use serde::{Deserialize,Serialize};
use std::{
	fs::File,io::{BufReader,BufWriter,Error as IOError,ErrorKind as IOErrorKind,Result as IOResult},marker::PhantomData,mem,path::Path
};
use token_dict::{TokenDict,Token,UTF8CharIter};
