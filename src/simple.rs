impl<B:Backend> BlockVariant<B> for Bias<B>{
	fn forward(&self,input:Value<B>)->Value<B>{input.map(|input:Tensor<B,2>|input+self.bias.val().unsqueeze(),None)}
	fn supports(&self,_encoding:u64)->bool{true}
	type BlockWith<C:Backend>=Bias<C>;
}
impl<B:Backend> BlockVariant<B> for Conv2D<B>{
	fn forward(&self,input:Value<B>)->Value<B>{
		if input.get_encoding()!=self.inputencoding{return input}
		input.map(|x:Tensor<B,4>|self.inner.forward(x),self.outputencoding)
	}
	fn supports(&self,encoding:u64)->bool{encoding==self.inputencoding}
	type BlockWith<C:Backend>=Conv2D<C>;
}
impl<B:Backend> BlockVariant<B> for Dense<B>{
	fn embed(&self,input:Tensor<B,2,Int>,inputclasses:usize,inputencoding:u64)->Value<B>{
		if !self.supports(inputencoding){return Value::unembedded(input,inputclasses,inputencoding)}
		let mut embedded=Embedding{weight:self.inner.weight.clone()}.forward(input);

		if let Some(b)=self.inner.bias.as_ref(){embedded=embedded+b.val().unsqueeze()};
		Value::new(embedded,self.outputencoding)
	}
	fn forward(&self,input:Value<B>)->Value<B>{
		if input.get_encoding()!=self.inputencoding{return input}
		input.map(|x:Tensor<B,2>|self.inner.forward(x),self.outputencoding)
	}
	fn supports(&self,encoding:u64)->bool{encoding==self.inputencoding}
	type BlockWith<C:Backend>=Dense<C>;
}
impl<B:Backend> BlockVariant<B> for Embed<B>{
	fn embed(&self,input:Tensor<B,2,Int>,inputclasses:usize,inputencoding:u64)->Value<B>{
		if !self.supports(inputencoding){return Value::unembedded(input,inputclasses,inputencoding)}
		Value::new(self.inner.forward(input),self.outputencoding)
	}
	fn forward(&self,input:Value<B>)->Value<B>{
		if input.get_encoding()!=self.inputencoding{return input}
		input.map(|x:Tensor<B,2>|Linear{bias:None,weight:self.inner.weight.clone()}.forward(x),self.outputencoding)
	}
	fn supports(&self,encoding:u64)->bool{encoding==self.inputencoding}
	type BlockWith<C:Backend>=Embed<C>;
}
impl<B:Backend> BlockVariant<B> for Detach<B>{
	fn forward(&self,input:Value<B>)->Value<B>{input.detach()}
	fn supports(&self,_encoding:u64)->bool{true}
	type BlockWith<C:Backend>=Detach<C>;
}
impl<B:Backend> BlockVariant<B> for Identity<B>{
	fn embed(&self,input:Tensor<B,2,Int>,inputclasses:usize,inputencoding:u64)->Value<B>{
		return Value::unembedded(input,inputclasses,inputencoding)
	}
	fn forward(&self,input:Value<B>)->Value<B>{input}
	fn supports(&self,_encoding:u64)->bool{true}
	type BlockWith<C:Backend>=Identity<C>;
}
impl<B:Backend> BlockVariant<B> for LayerNorm<B>{
	fn forward(&self,input:Value<B>)->Value<B>{input.map(|input:Tensor<B,2>|self.inner.forward(input),None)}
	fn supports(&self,_encoding:u64)->bool{true}
	type BlockWith<C:Backend>=LayerNorm<C>;
}
impl<B:Backend> BlockVariant<B> for MaxPool2D<B>{
	fn forward(&self,input:Value<B>)->Value<B>{
		if input.get_encoding()!=self.inputencoding{return input}
		input.map(|x:Tensor<B,4>|self.inner.forward(x),self.outputencoding)
	}
	fn supports(&self,encoding:u64)->bool{encoding==self.inputencoding}
	type BlockWith<C:Backend>=MaxPool2D<C>;
}
impl<B:Backend> BlockVariant<B> for RMSNorm<B>{
	fn forward(&self,input:Value<B>)->Value<B>{
		input.map(|x:Tensor<B,2>|{
			let rms=(x.clone().powf_scalar(2.0).mean_dim(1)+1.0E-6).sqrt();
			x/rms*self.gamma.val().unsqueeze()
		},None)
	}
	fn supports(&self,_encoding:u64)->bool{true}
	type BlockWith<C:Backend>=RMSNorm<C>;
}
impl<B:Backend> BlockVariant<B> for Relu<B>{
	fn embed(&self,input:Tensor<B,2,Int>,inputclasses:usize,inputencoding:u64)->Value<B>{Value::unembedded(input,inputclasses,inputencoding)}
	fn forward(&self,input:Value<B>)->Value<B>{input.map(|input:Tensor<B,1>|activation::relu(input),None)}
	fn supports(&self,_encoding:u64)->bool{true}
	type BlockWith<C:Backend>=Relu<C>;
}
impl<B:Backend> BlockVariant<B> for Tanh<B>{
	fn forward(&self,input:Value<B>)->Value<B>{input.map(|input:Tensor<B,1>|input.tanh(),None)}
	fn supports(&self,_encoding:u64)->bool{true}
	type BlockWith<C:Backend>=Tanh<C>;
}
impl<B:Backend> Bias<B>{
	/// get the number of features
	pub fn get_features(&self)->usize{self.bias.dims()[0]}
	/// create a new bias layer
	pub fn new(dimension:usize)->Self{
		Self{bias:Param::from_tensor(Tensor::zeros([dimension],&Default::default()))}
	}
}
impl<B:Backend> Conv2D<B>{
	/// creates a conv2d layer from the burn conv2d layer
	pub fn from_conv2d(inner:Conv2d<B>,inputencoding:u64,outputencoding:u64)->Self{
		Self{inner,inputencoding,outputencoding}
	}
	/// get the input encoding id
	pub fn get_input_encoding(&self)->u64{self.inputencoding}
	/// get the output encoding id
	pub fn get_output_encoding(&self)->u64{self.outputencoding}
	/// creates a new conv2d layer
	pub fn new(inputchannels:usize,inputencoding:u64,kernel:[usize;2],outputchannels:usize,outputencoding:u64,stride:[usize;2])->Self{
		Self{inner:Conv2dConfig::new([inputchannels,outputchannels],kernel).with_padding(PaddingConfig2d::Valid).with_stride(stride).init(&Default::default()),inputencoding,outputencoding}
	}
	/// sets the padding
	pub fn set_pad(&mut self,pad:[usize;2]){self.inner.padding=PaddingConfig2d::Explicit(pad[0],pad[1],pad[0],pad[1])}
	/// sets the padding
	pub fn with_pad(mut self,pad:[usize;2])->Self{
		self.set_pad(pad);
		self
	}
}
impl<B:Backend> Dense<B>{
	/// creates a dense layer from the burn linear layer
	pub fn from_linear(inner:Linear<B>,inputencoding:u64,outputencoding:u64)->Self{
		Self{inner,inputencoding,outputencoding}
	}
	/// get the input encoding id
	pub fn get_input_encoding(&self)->u64{self.inputencoding}
	/// get the number of input features
	pub fn get_input_features(&self)->usize{self.inner.weight.dims()[0]}
	/// get the output encoding id
	pub fn get_output_encoding(&self)->u64{self.outputencoding}
	/// get the number of output features
	pub fn get_output_features(&self)->usize{self.inner.weight.dims()[1]}
	/// creates a new dense layer
	pub fn new(inputencoding:u64,inputdimension:usize,outputencoding:u64,outputdimension:usize)->Self{
		Self{inner:LinearConfig::new(inputdimension,outputdimension).with_bias(false).init(&Default::default()),inputencoding,outputencoding}
	}
}
impl<B:Backend> Embed<B>{
	/// creates a dense layer from the burn embedding layer
	pub fn from_embedding(inner:Embedding<B>,inputencoding:u64,outputencoding:u64)->Self{
		Self{inner,inputencoding,outputencoding}
	}
	/// get the input encoding id
	pub fn get_input_encoding(&self)->u64{self.inputencoding}
	/// get the number of input features
	pub fn get_input_features(&self)->usize{self.inner.weight.dims()[0]}
	/// get the output encoding id
	pub fn get_output_encoding(&self)->u64{self.outputencoding}
	/// get the number of output features
	pub fn get_output_features(&self)->usize{self.inner.weight.dims()[1]}
	/// creates a new embedding layer
	pub fn new(inputencoding:u64,inputdimension:usize,outputencoding:u64,outputdimension:usize)->Self{
		Self{inner:EmbeddingConfig::new(inputdimension,outputdimension).init(&Default::default()),inputencoding,outputencoding}
	}
}
impl<B:Backend> Identity<B>{
	/// creates a trivial identity layer
	pub fn new()->Self{Default::default()}
}
impl<B:Backend> LayerNorm<B>{
	/// create from the burn layer
	pub fn from_layer_norm(inner:BurnLayerNorm<B>)->Self{
		Self{inner}
	}
	/// check if has bias
	pub fn has_bias(&self)->bool{self.inner.beta.is_some()}
	/// get the number of features
	pub fn get_features(&self)->usize{self.inner.gamma.dims()[0]}
	/// create a new layer norm block
	pub fn new(dim:usize,withbias:bool)->Self{
		Self{inner:LayerNormConfig::new(dim).with_bias(withbias).init(&Default::default())}
	}
}
impl<B:Backend> MaxPool2D<B>{
	/// creates pool layer
	pub fn from_max_pool_2d(inner:MaxPool2d,inputencoding:u64,outputencoding:u64)->Self{
		Self{inner,inputencoding,outputencoding,phantom:PhantomData}
	}
	/// get the input encoding id
	pub fn get_input_encoding(&self)->u64{self.inputencoding}
	/// get the output encoding id
	pub fn get_output_encoding(&self)->u64{self.outputencoding}
	/// creates a new conv2d layer
	pub fn new(inputencoding:u64,kernel:[usize;2],outputencoding:u64,stride:[usize;2])->Self{
		Self{inner:MaxPool2dConfig::new(kernel).with_padding(PaddingConfig2d::Valid).with_strides(stride).init(),inputencoding,outputencoding,phantom:PhantomData}
	}
	/// sets the padding
	pub fn set_pad(&mut self,pad:[usize;2]){self.inner.padding=PaddingConfig2d::Explicit(pad[0],pad[1],pad[0],pad[1])}
	/// sets the padding
	pub fn with_pad(mut self,pad:[usize;2])->Self{
		self.set_pad(pad);
		self
	}
}
impl<B:Backend> RMSNorm<B>{
	/// create from the param
	pub fn from_gamma(inner:Tensor<B,1>)->Self{
		Self{gamma:Param::from_tensor(inner)}
	}
	/// get the number of features
	pub fn get_features(&self)->usize{self.gamma.dims()[0]}
	/// create a new rms layer
	pub fn new(dimension:usize)->Self{
		Self{gamma:Param::from_tensor(Tensor::ones([dimension],&Default::default()))}
	}
}
impl<B:Backend> Relu<B>{
	/// creates a componentwise relu layer
	pub fn new()->Self{Default::default()}
}
impl<B:Backend> Tanh<B>{
	/// creates a componentwise tanh layer
	pub fn new()->Self{Default::default()}
}

#[derive(Debug,Deserialize,Module,Serialize)]
#[serde(bound="")]
/// layer that adds a constant
pub struct Bias<B:Backend>{
	#[serde(deserialize_with="data::deserialize_param")]
	#[serde(serialize_with="data::serialize_param")]
	bias:Param<Tensor<B,1>>,
}
#[derive(Debug,Deserialize,Module,Serialize)]
#[serde(bound="")]
/// 2d convolutional block
pub struct Conv2D<B:Backend>{
	#[serde(deserialize_with="data::deserialize_conv2d")]
	#[serde(serialize_with="data::serialize_conv2d")]
	inner:Conv2d<B>,
	inputencoding:u64,
	outputencoding:u64,
}
#[derive(Debug,Deserialize,Module,Serialize)]
#[serde(bound="")]
/// a simple block that applies a linear transformation
pub struct Dense<B:Backend>{
	#[serde(deserialize_with="data::deserialize_linear")]
	#[serde(serialize_with="data::serialize_linear")]
	inner:Linear<B>,
	inputencoding:u64,
	outputencoding:u64,
}
#[derive(Copy,Debug,Default,Deserialize,Module,Serialize)]
#[repr(transparent)]
/// detaches any values that pass through from the differentiation graph. supports all encodings; wrap in Adapt or Only if that isn't desired
pub struct Detach<B:Backend>{inner:PhantomData<B>}
#[derive(Debug,Deserialize,Module,Serialize)]
#[serde(bound="")]
/// a simple blocks that applies a linear transformation without bias, that can be efficiently used as an embedding
pub struct Embed<B:Backend>{
	#[serde(deserialize_with="data::deserialize_embedding")]
	#[serde(serialize_with="data::serialize_embedding")]
	inner:Embedding<B>,
	inputencoding:u64,
	outputencoding:u64,
}
#[derive(Copy,Debug,Default,Deserialize,Module,Serialize)]
#[repr(transparent)]
/// a trivial identity block. supports all encodings; wrap in Adapt or Only if that isn't desired
pub struct Identity<B:Backend>{inner:PhantomData<B>}
#[derive(Debug,Deserialize,Module,Serialize)]
#[repr(transparent)]
/// a layer norm block. supports all encoding; wrap in Adapt or Only if that isn't desired
pub struct LayerNorm<B:Backend>{
	#[serde(deserialize_with="data::deserialize_layer_norm")]
	#[serde(serialize_with="data::serialize_layer_norm")]
	inner:BurnLayerNorm<B>
}
#[derive(Debug,Deserialize,Module,Serialize)]
#[serde(bound="")]
/// 2d max pooling block
pub struct MaxPool2D<B:Backend>{
	#[serde(deserialize_with="data::deserialize_max_pool2d")]
	#[serde(serialize_with="data::serialize_max_pool2d")]
	inner:MaxPool2d,
	inputencoding:u64,
	outputencoding:u64,
	phantom:PhantomData<B>
}
#[derive(Debug,Deserialize,Module,Serialize)]
#[serde(bound="")]
pub struct RMSNorm<B:Backend>{
	#[serde(deserialize_with="data::deserialize_param")]
	#[serde(serialize_with="data::serialize_param")]
	gamma:Param<Tensor<B,1>>,
}
#[derive(Copy,Debug,Default,Deserialize,Module,Serialize)]
#[repr(transparent)]
/// a componentwise relu block. supports all encoding; wrap in Adapt or Only if that isn't desired
pub struct Relu<B:Backend>{inner:PhantomData<B>}
#[derive(Copy,Debug,Default,Deserialize,Module,Serialize)]
#[repr(transparent)]
/// a componentwise tanh block. supports all encodings; wrap in Adapt or Only if that isn't desired
pub struct Tanh<B:Backend>{inner:PhantomData<B>}

use burn::{
	module::Param,
	nn::{
		Embedding,
		EmbeddingConfig,
		LayerNorm as BurnLayerNorm,
		LayerNormConfig,
		Linear,
		LinearConfig,
		PaddingConfig2d,
		modules::{
			conv::{Conv2d,Conv2dConfig},pool::{MaxPool2d,MaxPool2dConfig}
		}
	},
	prelude::*,
	tensor::activation
};
use crate::data;
use serde::{Deserialize,Serialize};
use super::{BlockVariant,Value};
use std::marker::PhantomData;
