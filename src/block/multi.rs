impl Reduction{
	pub fn apply<B:Backend>(&self,values:impl IntoIterator<Item=Value<B>>)->Option<Value<B>>{
		fn cat_ranked<B:Backend,const N:usize>(meow:Vec<Value<B>>,dim:impl TryInto<isize>)->Option<Value<B>>{
			let dim=dim.try_into().map(|d|if d<0{N-(-d) as usize}else{d as usize}).ok().filter(|&d|d<N).expect("dim must be >=-N and <N");
			let encoding=meow.first()?.get_encoding();
			let loss=meow.iter().filter_map(Value::get_loss).reduce(|x,y|x+y);
			let mut m=Value::new(Tensor::cat(meow.into_iter().map(|x|x.get_data::<N>()).collect(),dim),encoding);

			if let Some(l)=loss{m.add_loss(l)}
			Some(m)
		}
		match self{
			&Self::Cat(dim)=>{// note: may need cat broadcast for reasons
				let mut r=0;  // TODO refactor logic into block and call from here
				let v:Vec<Value<B>>=values.into_iter().inspect(|v|r=r.max(v.get_rank())).collect();

				assert!(v.iter().is_sorted_by(|a,b|a.get_encoding()==b.get_encoding()));

				if v.len()==0{return None}
				match r{
					1=>cat_ranked::<B,1>(v,dim),
					2=>cat_ranked::<B,2>(v,dim),
					3=>cat_ranked::<B,3>(v,dim),
					4=>cat_ranked::<B,4>(v,dim),
					5=>cat_ranked::<B,5>(v,dim),
					6=>cat_ranked::<B,6>(v,dim),
					7=>cat_ranked::<B,7>(v,dim),
					8=>cat_ranked::<B,8>(v,dim),
					_=>panic!("expected rank between 1 and 8")
				}
			},
			Self::First=>values.into_iter().next(),
			Self::Sum=>values.into_iter().reduce(|x,y|x+y)
		}
	}
}
impl<A:AutodiffBackend<InnerBackend=B>,B:Backend,V:AutodiffModule<A,InnerModule=W>+BlockVariant<A>,W:BlockVariant<B>> AutodiffModule<A> for Branch<V>{
	fn from_inner(inner:Self::InnerModule)->Self{
		Branch{blocks:AutodiffModule::from_inner(inner.blocks),reduction:inner.reduction}
	}
	fn valid(&self)->Self::InnerModule{
		Branch{
			blocks:self.blocks.valid(),
			reduction:self.reduction
		}
	}
	type InnerModule=Branch<W>;
}
impl<A:AutodiffBackend<InnerBackend=B>,B:Backend,V:AutodiffModule<A,InnerModule=W>+BlockVariant<A>,W:BlockVariant<B>> AutodiffModule<A> for Sequential<V>{
	fn from_inner(inner:Self::InnerModule)->Self{Sequential(AutodiffModule::from_inner(inner.0))}
	fn valid(&self)->Self::InnerModule{self.0.valid().into()}
	type InnerModule=Sequential<W>;
}
impl<B:Backend,V:BlockVariant<B>> BlockVariant<B> for Branch<V>{
	fn clear(&mut self){self.blocks.iter_mut().for_each(V::clear)}
	fn detach_cache(&mut self){self.blocks.iter_mut().for_each(V::detach_cache)}
	fn embed(&self,input:Tensor<B,2,Int>,inputclasses:usize,inputencoding:u64)->Value<B>{
		self.reduction.apply(self.blocks.iter().filter(|b|b.supports(inputencoding)).map(|b|b.embed(input.clone(),inputclasses,inputencoding))).unwrap_or_else(||Value::unembedded(input,inputclasses,inputencoding))
	}
	fn embed_mut(&mut self,input:Tensor<B,2,Int>,inputclasses:usize,inputencoding:u64)->Value<B>{
		self.reduction.apply(self.blocks.iter_mut().filter(|b|b.supports(inputencoding)).map(|b|b.embed_mut(input.clone(),inputclasses,inputencoding))).unwrap_or_else(||Value::unembedded(input,inputclasses,inputencoding))
	}
	fn forward(&self,input:Value<B>)->Value<B>{
		let encoding=input.get_encoding();
		self.reduction.apply(self.blocks.iter().filter(|b|b.supports(encoding)).map(|b|b.forward(input.clone()))).unwrap_or(input)
	}
	fn forward_mut(&mut self,input:Value<B>)->Value<B>{
		let encoding=input.get_encoding();
		self.reduction.apply(self.blocks.iter_mut().filter(|b|b.supports(encoding)).map(|b|b.forward_mut(input.clone()))).unwrap_or(input)
	}
	fn supports(&self,encoding:u64)->bool{self.blocks.iter().map(|b|b.supports(encoding)).reduce(|x,y|x|y).unwrap_or(false)}
	type BlockWith<C:Backend>=Branch<V::BlockWith<C>>;
}
impl<B:Backend,V:BlockVariant<B>> BlockVariant<B> for Sequential<V>{
	fn clear(&mut self){self.0.iter_mut().for_each(V::clear)}
	fn detach_cache(&mut self){self.0.iter_mut().for_each(V::detach_cache)}
	fn embed(&self,input:Tensor<B,2,Int>,inputclasses:usize,inputencoding:u64)->Value<B>{
		if self.0.len()==0{return Value::unembedded(input,inputclasses,inputencoding)}
		let input=self.0[0].embed(input,inputclasses,inputencoding);
		self.0[1..].iter().fold(input,|x,b|b.forward(x))
	}
	fn embed_mut(&mut self,input:Tensor<B,2,Int>,inputclasses:usize,inputencoding:u64)->Value<B>{
		if self.0.len()==0{return Value::unembedded(input,inputclasses,inputencoding)}
		let input=self.0[0].embed_mut(input,inputclasses,inputencoding);
		self.0[1..].iter_mut().fold(input,|x,b|b.forward_mut(x))
	}
	fn forward(&self,input:Value<B>)->Value<B>{self.0.iter().fold(input,|x,b|b.forward(x))}
	fn forward_mut(&mut self,input:Value<B>)->Value<B>{self.0.iter_mut().fold(input,|x,b|b.forward_mut(x))}
	fn supports(&self,encoding:u64)->bool{self.0.iter().map(|b|b.supports(encoding)).reduce(|x,y|x|y).unwrap_or(false)}
	type BlockWith<C:Backend>=Sequential<V::BlockWith<C>>;
}
impl<B:Backend,V:BlockVariant<B>> Module<B> for Sequential<V>{
	fn collect_devices(&self,devices:Vec<B::Device>)->Vec<B::Device>{self.0.collect_devices(devices)}
	fn fork(self,device:&B::Device)->Self{Self(self.0.fork(device))}
	fn into_record(self)->Self::Record{self.0.into_record()}
	fn load_record(self,record:Self::Record)->Self{Self(self.0.load_record(record))}
	fn map<M:ModuleMapper<B>>(self,mapper:&mut M)->Self{Self(self.0.map(mapper))}
	fn to_device(self,device:&B::Device)->Self{Self(self.0.to_device(device))}
	fn visit<M:ModuleVisitor<B>>(&self,visitor:&mut M){self.0.visit(visitor)}
	type Record=<Vec<V> as Module<B>>::Record;
}
impl<B:Backend,V:BlockVariant<B>> Module<B> for Branch<V>{
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
	fn visit<M:ModuleVisitor<B>>(&self,visitor:&mut M){
		self.blocks.visit(visitor)
	}
	type Record=<Vec<V> as Module<B>>::Record;
}
impl<V:ModuleDisplay> ModuleDisplay for Branch<V>{}
impl<V:ModuleDisplay> ModuleDisplay for Sequential<V>{}
impl<V:ModuleDisplay> ModuleDisplayDefault for Branch<V>{
	fn content(&self,content:Content)->Option<Content>{self.blocks.content(content)}
}
impl<V:ModuleDisplay> ModuleDisplayDefault for Sequential<V>{
	fn content(&self,content:Content)->Option<Content>{self.0.content(content)}
}
impl<V> Default for Sequential<V>{
	fn default()->Self{Self(Vec::new())}
}
impl<V> From<RecursiveVariant<Sequential<V>>> for Sequential<V>{
	fn from(outer:RecursiveVariant<Sequential<V>>)->Self{*outer.0}
}
impl<V> From<Vec<V>> for Sequential<V>{
	fn from(inner:Vec<V>)->Self{Self(inner)}
}

#[derive(Clone,Copy,Debug,Default,Deserialize,Serialize)]
pub enum Reduction{
	Cat(i32),
	First,
	#[default]
	Sum
}
#[derive(Clone,Debug,Deserialize,Serialize)]
/// branch into multiple sub blocks, apply each to the input, then combine the results with a reduction
pub struct Branch<V>{blocks:Vec<V>,reduction:Reduction}
#[derive(Clone,Debug,Deserialize,Serialize)]
#[repr(transparent)]
/// wrapper for applying blocks in sequence. supports encodings at least one inner layer supports, so it's possible to use encoding ids to control which layers get executed
pub struct Sequential<V>(pub Vec<V>);

use burn::{
	module::{AutodiffModule,Content,ModuleDisplay,ModuleDisplayDefault,ModuleMapper,ModuleVisitor},
	prelude::*,
	tensor::backend::AutodiffBackend
};
use serde::{Deserialize,Serialize};
use super::{BlockVariant,RecursiveVariant,Value};
