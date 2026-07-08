impl ShareKey{
	/// create a share key from the inner data
	pub fn from_inner(generation:usize,lineage:u64,vtype:TypeId)->Self{
		Self{generation,lineage,vtype}
	}
	/// get the generation number
	pub fn get_generation(&self)->usize{self.generation}
	/// get the lineage id
	pub fn get_lineage(&self)->u64{self.lineage}
	/// gets the layer associated with this key if it exists
	pub fn get_shared<V:Any+Send>(&self)->Option<Shared<V>>{
		let handle=Shared{inner:Err(OnceCell::new()),key:*self};
		handle._try_upgrade_inner().map(|_|handle)
	}
	/// get the type id
	pub fn get_type(&self)->TypeId{self.vtype}
	/// create a new generation of a new lineage with the type
	pub fn new<V>()->Self{
		Self{
			generation:0,
			lineage:rand::random(),
			vtype:typeid::of::<V>()
		}
	}
	/// change type
	pub fn with_type<V>(mut self)->Self{
		self.vtype=typeid::of::<V>();
		self
	}
}
impl<'a,V:Any+DeserializeOwned+Send> Deserialize<'a> for Shared<V>{
	fn deserialize<D:Deserializer<'a>>(deserializer:D)->Result<Self,D::Error>{
		let (inner,generation,lineage):(Option<Arc<Mutex<V>>>,usize,u64)=Deserialize::deserialize(deserializer)?;
		let inner=inner.ok_or_else(OnceCell::new);
		let key=ShareKey{generation,lineage,vtype:typeid::of::<V>()};
		let maphandle=SHARE_MAP.get_or_init(Default::default);

		if let Some(layerhandle)=inner.as_ref().cloned().ok().or_else(||{
			let fallbackhandle=maphandle.get(&key.with_type::<ReserialFallback>())?;
			let fallbackvalue=fallbackhandle.downcast_ref::<ReserialFallback>()?()?;

			rmp_serde::from_slice(&fallbackvalue).map(|x|Arc::new(Mutex::new(x))).ok()
		}){
			maphandle.insert(key,Box::new(layerhandle));
		}
		let h=Self{inner,key};

		h._try_upgrade_inner();
		Ok(h)
	}
}
impl<A:AutodiffBackend<InnerBackend=B>,B:Backend,V:AutodiffModule<A,InnerModule=W>+BlockVariant<A>,W:BlockVariant<B>> AutodiffModule<A> for Clear<V>{
	fn from_inner(inner:Self::InnerModule)->Self{Clear(AutodiffModule::from_inner(inner.0))}
	fn valid(&self)->Self::InnerModule{Clear(self.0.valid())}
	type InnerModule=Clear<W>;
}
impl<A:AutodiffBackend<InnerBackend=B>,B:Backend,V:AutodiffModule<A,InnerModule=W>+BlockVariant<A>,W:BlockVariant<B>> AutodiffModule<A> for Registry<V>{
	fn from_inner(inner:Self::InnerModule)->Self{Registry(AutodiffModule::from_inner(inner.0))}
	fn valid(&self)->Self::InnerModule{Registry(self.0.valid())}
	type InnerModule=Registry<W>;
}
impl<A:AutodiffBackend<InnerBackend=B>,B:Backend,V:AutodiffModule<A,InnerModule=W>+BlockVariant<A>,W:BlockVariant<B>> AutodiffModule<A> for Shared<V>{
	fn from_inner(inner:Self::InnerModule)->Self{inner._derive(|v|V::from_inner(v.clone()),0)}
	fn valid(&self)   ->Self::InnerModule       {self ._derive(   V::valid,0)}
	type InnerModule=Shared<W>;
}
impl<A:AutodiffBackend<InnerBackend=B>,B:Backend,V:AutodiffModule<A,InnerModule=W>+BlockVariant<A>,W:BlockVariant<B>> AutodiffModule<A> for Update<V>{
	fn from_inner(inner:Self::InnerModule)->Self{Update(AutodiffModule::from_inner(inner.0))}
	fn valid(&self)->Self::InnerModule{Update(self.0.valid())}
	type InnerModule=Update<W>;
}
impl<B:Backend,V:BlockVariant<B>> BlockVariant<B> for Clear<V>{
	fn forward(&self,input:Value<B>)->Value<B>{
		self.0._do_layer(V::clear);
		input
	}
	fn forward_mut(&mut self,input:Value<B>)->Value<B>{
		self.0._do_layer(V::clear);
		input
	}
	fn supports(&self,_encoding:u64)->bool{true}
	type BlockWith<C:Backend>=Clear<V::BlockWith<C>>;
}
impl<B:Backend,V:BlockVariant<B>> BlockVariant<B> for Registry<V>{
	fn forward(&self,input:Value<B>)->Value<B>{input}
	fn supports(&self,_encoding:u64)->bool{true}
	type BlockWith<C:Backend>=Registry<V::BlockWith<C>>;
}
impl<B:Backend,V:BlockVariant<B>> BlockVariant<B> for Shared<V>{
	fn clear(&mut self){self._visit_mut(|l|l.clear())}
	fn detach_cache(&mut self){self._visit_mut(|l|l.detach_cache())}
	fn embed(&self,input:Tensor<B,2,Int>,inputclasses:usize,inputencoding:u64)->Value<B>{self._do_layer(|layer|layer.embed(input,inputclasses,inputencoding))}
	fn embed_mut(&mut self,input:Tensor<B,2,Int>,inputclasses:usize,inputencoding:u64)->Value<B>{self._do_layer(|layer|layer.embed_mut(input,inputclasses,inputencoding))}
	fn encoding_hint(&self)->Option<u64>{self._do_layer(|layer|layer.encoding_hint())}
	fn forward(&self,input:Value<B>)->Value<B>{self._do_layer(|layer|layer.forward(input))}
	fn forward_mut(&mut self,input:Value<B>)->Value<B>{self._do_layer(|layer|layer.forward_mut(input))}
	fn supports(&self,encoding:u64)->bool{self._do_layer(|layer|layer.supports(encoding))}
	type BlockWith<C:Backend>=Shared<V::BlockWith<C>>;
}
impl<B:Backend,V:BlockVariant<B>> BlockVariant<B> for Update<V>{
	fn clear(&mut self){self.0._visit_mut(|l|l.clear())}
	fn detach_cache(&mut self){self.0._visit_mut(|l|l.detach_cache())}
	fn embed(&self,input:Tensor<B,2,Int>,inputclasses:usize,inputencoding:u64)->Value<B>{self.0._do_layer(|layer|layer.embed_mut(input,inputclasses,inputencoding))}
	fn embed_mut(&mut self,input:Tensor<B,2,Int>,inputclasses:usize,inputencoding:u64)->Value<B>{self.0._do_layer(|layer|layer.embed_mut(input,inputclasses,inputencoding))}
	fn encoding_hint(&self)->Option<u64>{self.0._do_layer(|layer|layer.encoding_hint())}
	fn forward(&self,input:Value<B>)->Value<B>{self.0._do_layer(|layer|layer.forward_mut(input))}
	fn forward_mut(&mut self,input:Value<B>)->Value<B>{self.0._do_layer(|layer|layer.forward_mut(input))}
	fn supports(&self,encoding:u64)->bool{self.0._do_layer(|layer|layer.supports(encoding))}
	type BlockWith<C:Backend>=Update<V::BlockWith<C>>;
}
impl<B:Backend,V:BlockVariant<B>> Module<B> for Clear<V>{
	fn collect_devices(&self,devices:Vec<B::Device>)->Vec<B::Device>{self.0.collect_devices(devices)}
	fn fork(self,device:&B::Device)->Self{Self(self.0._map(|v|v.fork(device)))}
	fn into_record(self)->Self::Record{self.0.into_record()}
	fn load_record(self,record:Self::Record)->Self{Self(self.0.load_record(record))}
	fn map<M:ModuleMapper<B>>(self,mapper:&mut M)->Self{Self(self.0._map(|v|v.map(mapper)))}
	fn to_device(self,device:&B::Device)->Self{Self(self.0._map(|v|v.to_device(device)))}
	fn visit<M:ModuleVisitor<B>>(&self,visitor:&mut M){self.0.visit(visitor)}
	type Record=<(Option<V>,usize,u64) as Module<B>>::Record;
}
impl<B:Backend,V:BlockVariant<B>> Module<B> for Registry<V>{
	fn collect_devices(&self,devices:Vec<B::Device>)->Vec<B::Device>{self.0.collect_devices(devices)}
	fn fork(self,device:&B::Device)->Self{Self(self.0.fork(device))}
	fn into_record(self)->Self::Record{self.0.into_record()}
	fn load_record(self,record:Self::Record)->Self{Self(self.0.load_record(record))}
	fn map<M:ModuleMapper<B>>(self,mapper:&mut M)->Self{Self(self.0.map(mapper))}
	fn to_device(self,device:&B::Device)->Self{Self(self.0.to_device(device))}
	fn visit<M:ModuleVisitor<B>>(&self,visitor:&mut M){self.0.visit(visitor)}
	type Record=<Vec<Shared<V>> as Module<B>>::Record;
}
impl<B:Backend,V:BlockVariant<B>> Module<B> for Shared<V>{
	fn collect_devices(&self,mut devices:Vec<B::Device>)->Vec<B::Device>{
		self._visit(|v|devices=v.collect_devices(mem::take(&mut devices)));
		devices
	}
	fn fork(self,device:&B::Device)->Self{self._map(|v|v.fork(device))}
	fn into_record(self)->Self::Record{
		let mut layer=None;
		let key=self.key;

		self._visit(|v|layer=Some(v.clone()));
		(layer,key.generation,key.lineage).into_record()
	}
	fn load_record(self,record:Self::Record)->Self{
		let mut layer=None;
		let mut key=self.key;

		self._visit(|v|layer=Some(v.clone()));

		(layer,key.generation,key.lineage)=(layer,key.generation,key.lineage).load_record(record);

		let inner=layer.ok_or_else(OnceCell::new).map(|l|{
			let inner=Arc::new(Mutex::new(l));
			let maphandle=SHARE_MAP.get_or_init(Default::default);

			maphandle.insert(key,Box::from(inner.clone()));
			inner
		});
		Self{inner,key}
	}
	fn map<M:ModuleMapper<B>>(self,mapper:&mut M)->Self{self._map(|v|v.map(mapper))}
	fn to_device(self,device:&B::Device)->Self{self._map(|v|v.to_device(device))}
	fn visit<M:ModuleVisitor<B>>(&self,visitor:&mut M){self._visit(|v|v.visit(visitor))}
	type Record=<(Option<V>,usize,u64) as Module<B>>::Record;
}
impl<B:Backend,V:BlockVariant<B>> Module<B> for Update<V>{
	fn collect_devices(&self,devices:Vec<B::Device>)->Vec<B::Device>{self.0.collect_devices(devices)}
	fn fork(self,device:&B::Device)->Self{Self(self.0._map(|v|v.fork(device)))}
	fn into_record(self)->Self::Record{self.0.into_record()}
	fn load_record(self,record:Self::Record)->Self{Self(self.0.load_record(record))}
	fn map<M:ModuleMapper<B>>(self,mapper:&mut M)->Self{Self(self.0._map(|v|v.map(mapper)))}
	fn to_device(self,device:&B::Device)->Self{Self(self.0._map(|v|v.to_device(device)))}
	fn visit<M:ModuleVisitor<B>>(&self,visitor:&mut M){self.0.visit(visitor)}
	type Record=<(Option<V>,usize,u64) as Module<B>>::Record;
}
impl<B:Backend> BlockVariant<B> for Cache<B>{
	fn clear(&mut self){self.inner=None}
	fn detach_cache(&mut self){self.inner=self.inner.take().map(|x|x.detach())}
	fn forward(&self,input:Value<B>)->Value<B>{input}
	fn forward_mut(&mut self,input:Value<B>)->Value<B>{
		self.inner=Some(input.clone());
		input
	}
	fn supports(&self,_encoding:u64)->bool{true}
	type BlockWith<C:Backend>=Cache<C>;
}
impl<B:Backend> Cache<B>{
	/// get a clone of the inner value
	pub fn get_inner(&self)->Option<Value<B>>{self.inner.clone()}
	/// reference the inner value
	pub fn inner(&self)->&Option<Value<B>>{&self.inner}
	/// reference the inner value
	pub fn inner_mut(&mut self)->&mut Option<Value<B>>{&mut self.inner}
	/// convert into the inner value
	pub fn into_inner(self)->Option<Value<B>>{self.inner}
	/// create a new empty cache. use from(value) to create with an existing value inside
	pub fn new()->Self{
		Self{inner:None}
	}
}
impl<B:Backend> From<Option<Value<B>>> for Cache<B>{
	fn from(inner:Option<Value<B>>)->Self{
		Self{inner}
	}
}
impl<B:Backend> From<Value<B>> for Cache<B>{
	fn from(inner:Value<B>)->Self{Some(inner).into()}
}
impl<V:Any+Clone+Send> Clone for Clear<V>{
	fn clone(&self)->Self{Self(self.0.clone())}
}
impl<V:Any+Clone+Send> Clone for Registry<V>{
	fn clone(&self)->Self{Self(self.0.clone())}
}
impl<V:Any+Clone+Send> Clone for Shared<V>{
	fn clone(&self)->Self{self._derive(V::clone,1)}
}
impl<V:Any+Clone+Send> Clone for Update<V>{
	fn clone(&self)->Self{Self(self.0.clone())}
}
impl<V:Any+ModuleDisplay+Send> ModuleDisplay for Clear<V>{}
impl<V:Any+ModuleDisplay+Send> ModuleDisplay for Registry<V>{}
impl<V:Any+ModuleDisplay+Send> ModuleDisplay for Shared<V>{}
impl<V:Any+ModuleDisplay+Send> ModuleDisplay for Update<V>{}
impl<V:Any+ModuleDisplay+Send> ModuleDisplayDefault for Shared<V>{
	fn content(&self,content:Content)->Option<Content>{
		let mut c=None;
		self._visit(|l|c=l.content(content));

		c
	}
}
impl<V:Any+ModuleDisplay+Send> ModuleDisplayDefault for Clear<V>{
	fn content(&self,content:Content)->Option<Content>{self.0.content(content)}
}
impl<V:Any+ModuleDisplay+Send> ModuleDisplayDefault for Registry<V>{
	fn content(&self,content:Content)->Option<Content>{self.0.content(content)}
}
impl<V:Any+ModuleDisplay+Send> ModuleDisplayDefault for Update<V>{
	fn content(&self,content:Content)->Option<Content>{self.0.content(content)}
}
impl<V:Any+Send+Serialize> Serialize for Shared<V>{
	fn serialize<S:Serializer>(&self,serializer:S)->Result<S::Ok,S::Error>{
		let h=self.key;
		let fallback:Box<dyn Fn()->Option<Vec<u8>>+Send+Sync>=Box::new(move||{
			let h:Shared<V>=Shared{inner:Err(OnceCell::new()),key:h};
			h._try_upgrade_inner().and_then(|v|rmp_serde::to_vec(&*v.lock().ok()?).ok())
		});

		SHARE_MAP.get_or_init(Default::default).insert(self.key.with_type::<ReserialFallback>(),Box::new(fallback));
		(self.inner.as_ref().cloned().ok(),self.key.generation,self.key.lineage^RELINE.get()).serialize(serializer)
	}
}
impl<V:Any+Send> Clear<V>{
	/// create another non primary share with the same key
	pub fn share(&self)->Self{Self(self.0.share())}
	/// create another share with the same key, then swap it with self before returning, effectively taking the primary status of self and putting it in the returned value, leaving self non primary
	pub fn share_swap(&mut self)->Self{Self(self.0.share_swap())}
}
impl<V:Any+Send> From<V> for Clear<V>{
	fn from(inner:V)->Self{Self(inner.into())}
}
impl<V:Any+Send> From<V> for Shared<V>{
	fn from(inner:V)->Self{Self::new(inner)}
}
impl<V:Any+Send> From<V> for Update<V>{
	fn from(inner:V)->Self{Self(inner.into())}
}
impl<V:Any+Send> Shared<V>{
	/// derive a model from a reference
	fn _derive<F:FnOnce(&V)->U,U:Any+Send>(&self,f:F,inc:usize)->Shared<U>{
		let key=ShareKey{
			generation:self.key.generation+inc,
			lineage:self.key.lineage^RELINE.get(),
			vtype:typeid::of::<U>()
		};
		let inner=if let Some(x)=self._try_upgrade_inner(){
			let layer=x.lock().unwrap();
			let maphandle=SHARE_MAP.get_or_init(Default::default);
			let mut x=None;

			let x2=maphandle.entry(key).or_insert_with(||{
				let x1=Arc::new(Mutex::new(f(&*layer)));
				x=Some(x1.clone());

				Box::from(x1)
			});
			let x=x.unwrap_or_else(||x2.downcast_ref::<Arc<Mutex<U>>>().unwrap().clone());

			if self.is_primary(){Ok(x)}else{Err(Arc::downgrade(&x).into())}
		}else{
			//Err(OnceCell::new())
			panic!("An existing primary share should be alive at this point")
		};

		Shared{inner,key}
	}
	fn _do_layer<F:FnOnce(&mut V)->Y,Y>(&self,f:F)->Y{
		let inner=self._upgrade_inner();
		let mut lock=inner.lock().unwrap();

		f(&mut *lock)
	}
	/// map the inner value if this is a primary share
	fn _map<F:FnOnce(V)->V>(mut self,f:F)->Self where V:Clone{
		if let Ok(inner)=&mut self.inner{
			let mut layer=inner.lock().unwrap();
			*layer=f(layer.clone());
		}
		self
	}
	fn _try_upgrade_inner(&self)->Option<Arc<Mutex<V>>>{
		match &self.inner{
			Err(x)=>x.get().and_then(Weak::upgrade).or_else(||{
				let maphandle=SHARE_MAP.get_or_init(Default::default);
				let layerhandle=maphandle.get(&self.key)?;
				let layerhandle:&Arc<Mutex<V>>=layerhandle.downcast_ref()?;

				x.set(Arc::downgrade(layerhandle)).ok();
				Some(layerhandle.clone())
			}),
			Ok(x)=>Some(x.clone())
		}
	}
	fn _upgrade_inner(&self)->Arc<Mutex<V>>{self._try_upgrade_inner().expect("An existing primary share should be alive at this point")}
	/// visit the inner value if this is a primary share
	fn _visit<F:FnOnce(&V)>(&self,f:F){
		if let Ok(inner)=&self.inner{
			let layer=inner.lock().unwrap();
			f(&*layer);
		}
	}
	/// visit the inner value if this is a primary share
	fn _visit_mut<F:FnOnce(&mut V)>(&mut self,f:F){
		if let Ok(inner)=&self.inner{
			let mut layer=inner.lock().unwrap();
			f(&mut *layer);
		}
	}
	/// thin wraps Shared and uses its built in interior mutability to call clear when forward pass methods are called
	pub fn and_clear(self)->Clear<V>{Clear(self)}
	/// thin wraps Shared and uses its built in interior mutability to call mutable versions of functions even when shared versions are called
	pub fn and_update(self)->Update<V>{Update(self)}
	/// creates a new shared cache. convert a share to a clear to clear the cache on forward, and update to update the cache on forward
	pub fn cache<B:Backend>()->Self where V:From<Cache<B>>{Self::new(Cache::new().into())}
	/// get the share key
	pub fn get_key(&self)->ShareKey{self.key}
	/// make this share a primary share of its key. For correct map/visit/serial behavior, exactly one primary share should be included with each model, so this shouldn't be used unless a reference to the layer needs outside for serialization or something. Primary share status is preserved when cloning
	pub fn into_primary(mut self)->Self{
		self.make_primary();
		self
	}
	/// check if this is a primary share
	pub fn is_primary(&self)->bool{self.inner.is_ok()}
	/// make this share a primary share of its key. A previous primary share should still exist, otherwise the inner layer will have been dropped and the method will panic. For correct map/visit behavior, exactly one primary share should be included with each model, so this shouldn't be used unless a strong reference to the layer is needed outside, or the original primary is going to be dropped early. Primary share status is preserved when cloning
	pub fn make_primary(&mut self){self.inner=Ok(self._upgrade_inner())}
	/// create a new share from the inner layer. The result will be a 'primary' Shared reference that delegates to the inner module for mapping and visiting purposes, and its shares will be secondary shares referencing the same layer with the same key. For module map/visit methods to work correctly, exactly one primary share should be present per key per model. Primary share status is preserved when cloning
	pub fn new(inner:V)->Self{
		let inner=Arc::new(Mutex::new(inner));
		let key=ShareKey::new::<V>();
		let maphandle=SHARE_MAP.get_or_init(Default::default);

		maphandle.insert(key,Box::from(inner.clone()));
		Self{inner:Ok(inner.into()),key}
	}
	/// create another non primary share with the same key
	pub fn share(&self)->Self{
		let handle=Self{inner:Err(OnceCell::new()),key:self.key};
		handle._try_upgrade_inner();

		handle
	}
	/// create another share with the same key, then swap it with self before returning, effectively taking the primary status of self and putting it in the returned value, leaving self non primary. This can be useful when creating a collection of shares to avoid having to explicitly swap or take the original primary into somewhere
	pub fn share_swap(&mut self)->Self{
		let mut s=self.share();

		mem::swap(&mut s,self);
		s
	}
}
impl<V:Any+Send> Update<V>{
	/// create another non primary share with the same key
	pub fn share(&self)->Self{Self(self.0.share())}
	/// create another share with the same key, then swap it with self before returning, effectively taking the primary status of self and putting it in the returned value, leaving self non primary
	pub fn share_swap(&mut self)->Self{Self(self.0.share_swap())}
}
impl<V> Drop for Shared<V>{
	fn drop(&mut self){
		if let Some(layerhandle)=match mem::replace(&mut self.inner,Err(OnceCell::new())){
			Err(c)=>c.get().and_then(Weak::upgrade),
			Ok(a)=>Some(a)
		}{
			if Arc::strong_count(&layerhandle)==2&&Arc::weak_count(&layerhandle)==0&&let Some(maphandle)=SHARE_MAP.get(){
				maphandle.remove(&self.key);
				maphandle.remove(&self.key.with_type::<ReserialFallback>());
			}
		}
	}
}

/// make a clone of a model such that its shared layers have the same share pattern but with new lineages independent of the original. This function relies on thread local random number generation in a way that assumes clone implementations are single threaded, but it should still work so long as the Shares themselves are cloned by the calling thread.
pub fn break_lineage<V:Clone+Send>(module:impl AsRef<V>)->V{
	let reset=RELINE.get();
	RELINE.set(rand::random());

	let result=module.as_ref().clone();
	RELINE.set(reset);

	result
}

#[derive(Debug,Deserialize,Module,Serialize)]
#[repr(transparent)]
#[serde(bound="")]
/// layer that caches tensor values for potential reuse. intended to be shared and wrapped with update or clear. Supports all encodings. Wrap in Adapt or Only to limit encoding support
pub struct Cache<B:Backend>{inner:Option<Value<B>>}
#[derive(Debug,Deserialize,Serialize)]
#[repr(transparent)]
#[serde(bound(deserialize="V:Any+DeserializeOwned+Send",serialize="V:Any+Send+Serialize"))]
/// layer that clears the cache of shared layer through interior mutability. This just thin wraps Shared and uses its built in interior mutability to call clear_cache whenever a forward pass method is called. Supports all encodings. Wrap in Adapt or Only to limit encoding support
pub struct Clear<V>(pub Shared<V>);
#[derive(Debug,Deserialize,Serialize)]
#[repr(transparent)]
#[serde(bound(deserialize="V:Any+DeserializeOwned+Send",serialize="V:Any+Send+Serialize"))]
/// a place to put primary shares in a model that lacks obvious blocks to keep them in. This is effectively an identity block with a payload
pub struct Registry<V>(pub Vec<Shared<V>>);
#[derive(Clone,Copy,Debug,Eq,Hash,PartialEq)]
/// the key type for identifying shares
pub struct ShareKey{generation:usize,lineage:u64,vtype:TypeId}
#[derive(Debug)]
/// wraps the inner layer to allow parameter sharing. Each Share is identified by a key storing its generation and lineage, and shares with the same key reference the same layer. Multiple shares with the same generation and lineage but different reserializable vtypes may coexist due to change_backend, deserialize, or valid, but exactly which share the data of yet another such share will be derived from is unspecified. Creating a Share with new creates a new lineage, and cloning increments the generation. Use share::break_lineage to create a clone of a model with new lineages for shared layers that are associated with each other but not with descendants of the original model. Some shares may be considered 'primary'. On modules mapping/serialization/visiting only the primary shares will delegate those to the inner layer. To minimize unexpected behavior, I recommend keeping shares with the same key within the same model, and breaking lineage whenever the Shared layers of one instance or model shouldn't be shared with another model.
pub struct Shared<V>{inner:Result<Arc<Mutex<V>>,OnceCell<Weak<Mutex<V>>>>,key:ShareKey}
#[derive(Debug,Deserialize,Serialize)]
#[repr(transparent)]
#[serde(bound(deserialize="V:Any+DeserializeOwned+Send",serialize="V:Any+Send+Serialize"))]
/// layer that updates a shared layer through interior mutability. This just thin wraps Shared and uses its built in interior mutability to call mutable versions of functions even when shared versions are called
pub struct Update<V>(pub Shared<V>);

/// map share keys to share info
static SHARE_MAP:OnceLock<DashMap<ShareKey,Box<dyn Any+Send+Sync>>>=OnceLock::new();
thread_local!{
	static RELINE:Cell<u64>=const{Cell::new(0)};
}

type ReserialFallback=Box<dyn Fn()->Option<Vec<u8>>+Send+Sync>;

use burn::{
	module::{AutodiffModule,Content,ModuleDisplay,ModuleDisplayDefault,ModuleMapper,ModuleVisitor},
	prelude::*,
	tensor::backend::AutodiffBackend
};
use dashmap::DashMap;
use serde::{Deserialize,Deserializer,Serialize,Serializer,de::DeserializeOwned};
use super::{BlockVariant,Value};
use std::{
	any::{Any,TypeId},cell::{Cell,OnceCell},mem,sync::{Arc,Mutex,OnceLock,Weak}
};
